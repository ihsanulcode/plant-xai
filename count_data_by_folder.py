import os

def count_and_print_folder_data(root_dir):
    if not os.path.isdir(root_dir):
        print(f"Directory not found: {root_dir}")
        return

    print(f"\nDataset: {root_dir}")
    total = 0

    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)

        if os.path.isdir(folder_path):
            count = sum(
                1 for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
            )
            total += count
            print(f"{folder_name}: {count}")

    print(f"Total files: {total}")


def main():
    # change paths if needed
    train_dir = "dataset/train"
    valid_dir = "dataset/valid"

    count_and_print_folder_data(train_dir)
    count_and_print_folder_data(valid_dir)


if __name__ == "__main__":
    main()
