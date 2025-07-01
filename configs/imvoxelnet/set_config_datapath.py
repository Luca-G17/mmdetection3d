import sys

def main():
    config_path = sys.argv[1]
    data_root = sys.argv[2]
    with open(config_path, "a") as cfg:
        data = cfg.readlines()
        data[0] = f"data_root = '{data_root}'"

if __name__ == "__main__":
    main()