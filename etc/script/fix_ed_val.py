def __fix_ed_impl(ed_text):
    """ Fix the text value of ED.
    """
    ed = {
        'Infinity': 0,
        '1.0': 1,
        '0.5': 2,
        '0.33333334': 3,
        '0.0': 4
        }.get(ed_text, -1)
    if ed == -1:
        raise ValueError(ed_text)
    return (4 - ed) / 4.0


def fix_ed_val(in_path, out_path):
    with open(out_path, 'w') as out_file:
        with open(in_path, 'r') as in_file:
            is_feature_line = True
            curr_error = None
         
            for line in in_file:
                line = line.strip()
                
                # Omit feature lines.
                if is_feature_line:
                    if len(line) == 0:
                        is_feature_line = False
                
                # Find and fix candidate lines.
                else:
                    fields = line.split('\t')
                    if len(fields) >= 1:
                        if fix_ed_val:
                            fields[1] = str(fix_ed(fields[1]))
                    line = '\t'.join(fields)
                
                # Write to the output file.
                out_file.write(line)
    

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print 'Usage: python %s SRC_FILE DIST_FILE'
    else:
        fix_ed_val(sys.argv[1], sys.argv[2])

