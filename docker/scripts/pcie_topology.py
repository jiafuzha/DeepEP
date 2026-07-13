#!/usr/bin/env python3
"""Discover and validate XPU/RDMA groupings that share one PCIe switch."""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


BDF_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = SCRIPT_DIR.parent / ".env.compose"


class TopologyError(RuntimeError):
    """Raised when the expected PCIe topology is not present."""


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def numeric_suffix(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    if not match:
        raise TopologyError(f"Cannot extract numeric suffix from {name!r}")
    return int(match.group(1))


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def bridge_chain(endpoint_bdf: str) -> list[str]:
    device_path = (Path("/sys/bus/pci/devices") / endpoint_bdf).resolve()
    chain: list[str] = []

    for parent in device_path.parents:
        parent_name = parent.name
        if not BDF_RE.match(parent_name):
            continue
        class_path = Path("/sys/bus/pci/devices") / parent_name / "class"
        if class_path.exists() and read_text(class_path).startswith("0x0604"):
            chain.append(parent_name)

    return chain


def switch_anchor(endpoint_bdf: str) -> str | None:
    chain = bridge_chain(endpoint_bdf)
    if len(chain) < 2:
        return None
    return chain[-2]


def describe_device(kind: str, name: str) -> dict[str, str | list[str] | None]:
    class_dir = Path("/sys/class/drm" if kind == "render" else "/sys/class/infiniband")
    symlink = class_dir / name
    if not symlink.exists():
        raise TopologyError(f"{kind} device {name!r} does not exist in {class_dir}")

    device_path = (symlink / "device").resolve()
    endpoint_bdf = device_path.name
    if not BDF_RE.match(endpoint_bdf):
        raise TopologyError(f"Resolved device path for {name!r} is not a PCI endpoint: {device_path}")

    return {
        "name": name,
        "kind": kind,
        "endpoint_bdf": endpoint_bdf,
        "switch_anchor": switch_anchor(endpoint_bdf),
        "bridge_chain": bridge_chain(endpoint_bdf),
    }


def list_devices(kind: str) -> list[dict[str, str | list[str] | None]]:
    pattern = "renderD*" if kind == "render" else "mlx5_*"
    class_dir = Path("/sys/class/drm" if kind == "render" else "/sys/class/infiniband")
    names = sorted((path.name for path in class_dir.glob(pattern)), key=numeric_suffix)
    return [describe_device(kind, name) for name in names]


def group_candidates() -> dict[str, dict[str, list[dict[str, str | list[str] | None]]]]:
    grouped: dict[str, dict[str, list[dict[str, str | list[str] | None]]]] = defaultdict(
        lambda: {"render": [], "mlx5": []}
    )

    for render in list_devices("render"):
        anchor = render["switch_anchor"]
        if anchor:
            grouped[str(anchor)]["render"].append(render)

    for hca in list_devices("mlx5"):
        anchor = hca["switch_anchor"]
        if anchor:
            grouped[str(anchor)]["mlx5"].append(hca)

    for anchor_data in grouped.values():
        anchor_data["render"].sort(key=lambda item: numeric_suffix(str(item["name"])))
        anchor_data["mlx5"].sort(key=lambda item: numeric_suffix(str(item["name"])))

    return dict(grouped)


def render_ordinals(render_names: Iterable[str]) -> str:
    global_renders = [str(item["name"]) for item in list_devices("render")]
    ordinal_map = {name: index for index, name in enumerate(global_renders)}
    ordinals = [str(ordinal_map[name]) for name in render_names]
    return ",".join(ordinals)


def choose_nodes(
    groups: dict[str, dict[str, list[dict[str, str | list[str] | None]]]],
    nodes: int,
    render_per_node: int,
    mlx5_per_node: int,
) -> list[dict[str, str]]:
    eligible_distinct = [
        (anchor, devices)
        for anchor, devices in groups.items()
        if len(devices["render"]) >= render_per_node and len(devices["mlx5"]) >= mlx5_per_node
    ]
    eligible_distinct.sort(
        key=lambda item: (
            len(item[1]["render"]),
            len(item[1]["mlx5"]),
            item[0],
        ),
        reverse=True,
    )

    if len(eligible_distinct) >= nodes:
        chosen = eligible_distinct[:nodes]
        return [
            build_node_config(
                anchor=anchor,
                render_names=[str(item["name"]) for item in devices["render"][:render_per_node]],
                mlx5_names=[str(item["name"]) for item in devices["mlx5"][:mlx5_per_node]],
            )
            for anchor, devices in chosen
        ]

    for anchor, devices in eligible_distinct:
        if len(devices["render"]) >= nodes * render_per_node and len(devices["mlx5"]) >= nodes * mlx5_per_node:
            node_configs: list[dict[str, str]] = []
            for node_index in range(nodes):
                render_start = node_index * render_per_node
                mlx5_start = node_index * mlx5_per_node
                node_configs.append(
                    build_node_config(
                        anchor=anchor,
                        render_names=[
                            str(item["name"])
                            for item in devices["render"][render_start : render_start + render_per_node]
                        ],
                        mlx5_names=[
                            str(item["name"])
                            for item in devices["mlx5"][mlx5_start : mlx5_start + mlx5_per_node]
                        ],
                    )
                )
            return node_configs

    raise TopologyError(
        "Could not find enough PCIe-switch-local resources for the requested layout. "
        f"Need {nodes} node(s) with {render_per_node} render nodes and {mlx5_per_node} mlx5 HCAs each.\n"
        + summarize_groups(groups)
    )


def build_node_config(anchor: str, render_names: list[str], mlx5_names: list[str]) -> dict[str, str]:
    return {
        "switch_anchor": anchor,
        "render_names": ",".join(render_names),
        "ze_affinity_mask": render_ordinals(render_names),
        "mlx5_names": ",".join(mlx5_names),
        "ucx_net_devices": ",".join(f"{name}:1" for name in mlx5_names),
    }


def summarize_groups(groups: dict[str, dict[str, list[dict[str, str | list[str] | None]]]]) -> str:
    lines = ["Discovered PCIe-switch candidates:"]
    for anchor in sorted(groups):
        renders = ",".join(str(item["name"]) for item in groups[anchor]["render"]) or "-"
        mlx5s = ",".join(str(item["name"]) for item in groups[anchor]["mlx5"]) or "-"
        lines.append(f"  {anchor}: render=[{renders}] mlx5=[{mlx5s}]")
    return "\n".join(lines)


def write_env_file(
    output_path: Path,
    node_configs: list[dict[str, str]],
    nodes: int,
    render_per_node: int,
    mlx5_per_node: int,
) -> None:
    lines = [
        "# Auto-generated by docker/scripts/pcie_topology.py discover-env",
        "# Host networking is used on purpose so the RDMA HCAs remain visible in both containers.",
        "IMAGE_NAME=deepep_jiafuzha",
        "WORKSPACE_ROOT=/root/jiafuzha",
        "MASTER_ADDR=127.0.0.1",
        "MASTER_PORT=29500",
        f"NNODES={nodes}",
        f"NPROC_PER_NODE={render_per_node}",
        "TORCH_BACKEND=ccl",
        "CCL_ATL_TRANSPORT=ofi",
        "FI_PROVIDER=verbs;ofi_rxm",
        "CCL_LOG_LEVEL=info",
        "FI_LOG_LEVEL=warn",
        f"EXPECTED_RENDER_PER_NODE={render_per_node}",
        f"EXPECTED_MLX5_PER_NODE={mlx5_per_node}",
    ]

    for index, config in enumerate(node_configs):
        prefix = f"NODE{index}"
        lines.extend(
            [
                f"{prefix}_HOSTNAME=node{index}",
                f"{prefix}_CONTAINER_NAME=sim-node{index}",
                f"{prefix}_SWITCH_ANCHOR={config['switch_anchor']}",
                f"{prefix}_RENDER_NODES={config['render_names']}",
                f"{prefix}_ZE_AFFINITY_MASK={config['ze_affinity_mask']}",
                f"{prefix}_MLX5_HCAS={config['mlx5_names']}",
                f"{prefix}_UCX_NET_DEVICES={config['ucx_net_devices']}",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_selection(
    render_nodes: list[str],
    mlx5_hcas: list[str],
    expected_switch: str | None,
    expected_render_count: int | None,
    expected_mlx5_count: int | None,
) -> None:
    if expected_render_count is not None and len(render_nodes) != expected_render_count:
        raise TopologyError(
            f"Expected {expected_render_count} render nodes but got {len(render_nodes)}: {','.join(render_nodes)}"
        )
    if expected_mlx5_count is not None and len(mlx5_hcas) != expected_mlx5_count:
        raise TopologyError(
            f"Expected {expected_mlx5_count} mlx5 HCAs but got {len(mlx5_hcas)}: {','.join(mlx5_hcas)}"
        )

    devices = [describe_device("render", name) for name in render_nodes] + [
        describe_device("mlx5", name) for name in mlx5_hcas
    ]
    missing_anchor = [str(device["name"]) for device in devices if device["switch_anchor"] is None]
    if missing_anchor:
        raise TopologyError(
            "Selected devices are not behind a detectable PCIe switch anchor: "
            + ",".join(missing_anchor)
        )
    anchors = {str(device["switch_anchor"]) for device in devices if device["switch_anchor"]}

    if len(anchors) != 1:
        details = "\n".join(
            f"  {device['name']} -> endpoint {device['endpoint_bdf']} -> switch {device['switch_anchor']}"
            for device in devices
        )
        raise TopologyError(
            "Selected devices do not share one PCIe switch anchor.\n"
            + details
        )

    anchor = next(iter(anchors))
    if expected_switch and anchor != expected_switch:
        raise TopologyError(f"Selected devices resolve to switch {anchor}, expected {expected_switch}")

    print(f"Validated shared PCIe switch {anchor}")
    for device in devices:
        chain = " -> ".join(str(item) for item in device["bridge_chain"])
        print(f"  {device['name']}: endpoint={device['endpoint_bdf']} bridges={chain}")


def cmd_discover_env(args: argparse.Namespace) -> int:
    groups = group_candidates()
    node_configs = choose_nodes(
        groups=groups,
        nodes=args.nodes,
        render_per_node=args.render_per_node,
        mlx5_per_node=args.mlx5_per_node,
    )
    output_path = Path(args.output).resolve()
    write_env_file(
        output_path=output_path,
        node_configs=node_configs,
        nodes=args.nodes,
        render_per_node=args.render_per_node,
        mlx5_per_node=args.mlx5_per_node,
    )
    print(f"Wrote {output_path}")
    print(summarize_groups(groups))
    for index, config in enumerate(node_configs):
        print(
            f"node{index}: switch={config['switch_anchor']} "
            f"render=[{config['render_names']}] mlx5=[{config['mlx5_names']}] "
            f"ZE_AFFINITY_MASK={config['ze_affinity_mask']}"
        )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    validate_selection(
        render_nodes=split_csv(args.render_nodes),
        mlx5_hcas=split_csv(args.mlx5_hcas),
        expected_switch=args.expected_switch,
        expected_render_count=args.expected_render_count,
        expected_mlx5_count=args.expected_mlx5_count,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_env = subparsers.add_parser(
        "discover-env",
        help="Discover two node-sized groups of XPUs and mlx5 HCAs that share a PCIe switch.",
    )
    discover_env.add_argument("--nodes", type=int, default=2)
    discover_env.add_argument("--render-per-node", type=int, default=4)
    discover_env.add_argument("--mlx5-per-node", type=int, default=4)
    discover_env.add_argument("--output", default=str(DEFAULT_ENV_PATH))
    discover_env.set_defaults(func=cmd_discover_env)

    validate = subparsers.add_parser(
        "validate-selection",
        help="Validate that the selected render nodes and mlx5 HCAs share one PCIe switch.",
    )
    validate.add_argument("--render-nodes", required=True)
    validate.add_argument("--mlx5-hcas", required=True)
    validate.add_argument("--expected-switch")
    validate.add_argument("--expected-render-count", type=int)
    validate.add_argument("--expected-mlx5-count", type=int)
    validate.set_defaults(func=cmd_validate)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except TopologyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
