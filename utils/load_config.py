from __future__ import absolute_import
from __future__ import print_function

import sys
import json
import argparse

from collections import namedtuple


def load_configuration(configuration_file):
    with open(configuration_file, 'r') as content_file:
        content = content_file.read()

    return json.loads(content, object_hook=lambda d: namedtuple('Configuration', d.keys())(*d.values()))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config_file', help='Path to configuraton file', required=True)

    args = parser.parse_args()

    config = load_configuration(args.config_file)

    print(config)



if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))