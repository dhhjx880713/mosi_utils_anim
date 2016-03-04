
_lines = []
_active = True


def activate():
    global _active
    _active = True


def deactivate():
    global _active
    _active = False


def write_log(*args):
    global _active
    global _lines
    if _active:
        line = " ".join(map(str, args))
        print line
        _lines.append(line)


def save_log(filename):
    global _lines
    with open(filename, "wb") as outfile:
        for l in _lines:
            outfile.write(l+"\n")


def clear_log():
    global _lines
    _lines = []