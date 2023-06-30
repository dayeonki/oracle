def loadlines(path):
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    return lines


def loadlines_multifile(paths):
    lines = []
    for path in paths:
        lines += loadlines(path)
    return lines


def sort_files(files):
    return sorted(files)


def equal_length(obj1, obj2):
    return len(obj1) == len(obj2)
