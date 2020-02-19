
def get_csv_and_pred(i=42):
    elements = ['Pr', 'Ni', 'Ru', 'Ne', 'Rb', 'Pt', 'La', 'Na', 'Nb', 'Nd',
            'Mg', 'Li', 'Pb', 'Re', 'Tl', 'Lu', 'Pd', 'Ti', 'Te', 'Rh',
            'Tc', 'Sr', 'Ta', 'Be', 'Ba', 'Tb', 'Yb', 'Si', 'Bi', 'W',
            'Gd', 'Fe', 'Br', 'Dy', 'Hf', 'Hg', 'Y', 'He', 'C', 'B', 'P',
            'F', 'I', 'H', 'K', 'Mn', 'O', 'N', 'Kr', 'S', 'U', 'Sn', 'Sm',
            'V', 'Sc', 'Sb', 'Mo', 'Os', 'Se', 'Th', 'Zn', 'Co', 'Ge',
            'Ag', 'Cl', 'Ca', 'Ir', 'Al', 'Ce', 'Cd', 'Ho', 'As', 'Ar',
            'Au', 'Zr', 'Ga', 'In', 'Cs', 'Cr', 'Tm', 'Cu', 'Er']
    np.random.seed(seed=i)
    elem_rnd_chem = {}
    size=1200
    for elem in elements:
        elem_rnd_chem[elem] = np.random.normal(size=size)
    df_rnd_chem = pd.DataFrame(elem_rnd_chem).T
    csv_name = 'random_{:0.0f}.csv'.format(size)
    df_rnd_chem.to_csv(csv_name, index_label='element')

if __name__ == '__main__':
    get_csv_and_pred(i=42)