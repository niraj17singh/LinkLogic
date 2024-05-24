from warnings import warn


def get_key_value_pair_from_string(s):

    key = None
    value = None

    if ': ' in s:
        toks = s.split(': ')
        key = toks[0]
        value = ''.join(toks[1:])
        value = value.rstrip()
    else:
        warn(f'Unexpected string value: {s}')

    return key, value


def get_mesh_to_disease_ontology(do_file = '../data/disease_ontology/doid.obo.txt'):

    #do_to_mesh = dict()
    mesh_to_do = dict()

    counter = 0
    with open(do_file, 'r') as f:

        # burn through header
        line = f.readline()
        while line != '[Term]\n':
            line = f.readline()

        # go through each term
        while line:

            line = f.readline()

            counter += 1
#            if counter % 100 == 0:
#                print(f'counter={counter}')

            do_ids = []
            mesh_ids = []
            while line not in ['[Term]\n', '', '[Typedef]\n']:# and line != '':

                key, value = get_key_value_pair_from_string(line)

                if key == 'id' or key == 'alt_id':
                    do_ids.append(value)

                if key == 'xref' and value.startswith('MESH'):
                    #print('appending mesh id')
                    mesh_ids.append(value)

                line = f.readline()
            if len(do_ids) >= 1 and len(mesh_ids) >= 1:
                #do_to_mesh[do_ids[0]] = mesh_ids
                for mesh in mesh_ids:
                    mesh_to_do[mesh] = do_ids[0]
    return mesh_to_do
    #print(do_to_mesh)

if __name__ == '__main__':
    mesh_to_do = get_mesh_to_disease_ontology()
