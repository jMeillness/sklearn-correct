from os import path
import pkg_resources
import random
import string


def random_string(str_len=0):
    """ Generate a string, which characters are randomly selected from the
    English lowercase character set.

    Args:
        str_len: a integer indicates the length of the output string. Giving a
            non-positive value will indicates a random length no longer than 10.
            (default: 0)

    Returns:
        A randomly generated English string.
    """
    if str_len < 1:
        str_len = random.randint(1, 10)
    return ''.join(random.choice(string.lowercase) for i in range(str_len))


def random_strings(num, str_len=0):
    """ Generate a list of strings, which characters are randomly selected from
    the English lowercase character set.

    Args:
        str_len: a integer indicates the length of the output string. Giving a
            non-positive value will indicates a random length no longer than 10.
            (default: 0)
        num: a integer indicates the number of strings to be generated.

    Returns:
        A list of randomly generated English strings.
    """
    str_list = []
    while len(str_list) < num:
        new_str = random_string(str_len)
        if new_str not in str_list:
            str_list.append(new_str)
    return str_list
    


def get_files(directory, regex):
    """ Get files in directory.

    Args:
        directory: a string indicates the pathname to a directory in the file
            system.
        regex: a regular expression used to filter the files by file name.

    Returns:
        a list of file pathnames.
    """
    return sorted(glob.glob(path.join(directory, regex)))


TEST_RESOURCE_DIR = pkg_resources.resource_filename('sklc.test', 'resources')


class ResourceError(Exception):
    """ Raises when intended resource do not exists.
    """
    pass


def get_resource_file(pathname, regex='*'):
    """ Get file(s) in the test resource directory.

    Args:
        pathname: a string indicates the relative pathname in the test resource
            directory.
        regex: a regular expression used to filter the files by file name.

    Returns:
        A file to the pathname if the given pathname is a file. Or a list of
        files if the given pathname is a diretory.
    """
    resource_path = path.join(TEST_RESOURCE_DIR, pathname)
    if not path.exists(resource_path):
        raise ResourceError('Resource \'%s\' is not found in \'%s\'.' %
                (pathname, TEST_RESOURCE_DIR))
    if path.isfile(resource_path):
        return resource_path
    elif path.isdir(resource_path):
        return get_files(resource_path, regex)
