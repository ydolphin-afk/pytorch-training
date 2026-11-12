from pathlib import Path

if __name__ == "__main__":
    #1
    data_dir = "04/data"
    dir_path = Path(data_dir).resolve()
    print(dir_path)
    #2
    file_list = list(dir_path.glob("*"))
    for dir in enumerate(file_list):
        print(dir[1])
