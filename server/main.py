import sys

from server.config import load_nodes_config
from server.coordinator import Coordinator


def main():
    config_path = "server/nodes.json"
    if len(sys.argv) >= 2:
        config_path = sys.argv[1]

    nodes = load_nodes_config(config_path)

    coord = Coordinator(nodes)
    coord.discover_nodes()
    coord.print_summary()


if __name__ == "__main__":
    main()
