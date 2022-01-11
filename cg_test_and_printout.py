#!/usr/bin/env python3

######################################################################
#                ______                  ___                         #
#               / ____/___  __  ______  / (_)___  ____ _             #
#              / /   / __ \/ / / / __ \/ / / __ \/ __ `/             #
#             / /___/ /_/ / /_/ / /_/ / / / / / / /_/ /              #
#             \____/\____/\__,_/ .___/_/_/_/ /_/\__, /               #
#                             /_/              /____/                #
#                                                                    #
######################################################################

# This script checks the coupling coefficents in qdef, and generates a
# LaTeX  output.  To  fully  compile  the  LaTeX output use the string
# "template" here included. Modify the outfile variable accordingly.

# The   check   consists   in  determining  if  the  unitary  matrices
# corresponding  to coupling coefficients adequately block-diagonalize
# the  tensor  products of the irreducible representation matrices for
# the group generators.

from qdef import *
from sympy.physics.quantum import TensorProduct

outfile = '/Users/juan/Library/Mobile Documents/com~apple~CloudDocs/iCloudFiles/Theoretical Division/CGmatrices_long.tex'

template = r'''\documentclass[10pt]{article}
\usepackage{multicol}
\usepackage{fontawesome}
\usepackage{titlesec}
\usepackage[svgnames]{xcolor}
\usepackage{booktabs}
\usepackage{tocloft}
\setlength{\cftsubsecnumwidth}{4em}

\setlength\columnsep{50pt}
\usepackage[top=2in,bottom=2in,right=1in,left=1in,paperwidth=19in,paperheight=31.29in]{geometry}
\usepackage{amsmath}

\usepackage{hyperref}
\hypersetup{colorlinks, 
    citecolor=black,
    filecolor=black,
    linkcolor=DarkMagenta,
    urlcolor=black
}

\titleformat{\section}[hang]{\normalfont\Huge\bfseries\filcenter}{\thesection}{1em}{}
\titleformat{\subsection}[hang]{\normalfont\Large\bfseries\filcenter}{\thesubsection}{1em}{}
\titleformat{\subsubsection}[hang]{\normalfont\large\bfseries\filcenter}{\thesubsubsection}{1em}{}

\begin{document}

\title{\Huge Coupling Coefficients}
\maketitle

\begin{multicols}{4}
\tableofcontents
\end{multicols}
\newpage

\input{CGmatrices_long.tex}

\end{document}
'''

def format_latex_matrix(mat, col_widths, row_widths, col_labels):
    dividers = 'cc|'
    for col_width in col_widths:
        dividers += 'c'*col_width
        dividers += '|'
    dividers = dividers[:-1]
    phantom_formater = '{%s}' % ('c' * mat.cols)
    lat = sp.latex(mat)
    if 'matrix' in lat:
        lat = lat.replace('\\begin{matrix}', '\\begin{array}{%s}\n' % dividers)
    else:
        dividers = '{%s}\n' % dividers
        lat = lat.replace(phantom_formater, dividers)
    lat = lat.replace('\\left[','').replace('\\right]','')
    if 'matrix' in lat:
        lat = lat.replace('\\end{matrix}','\n\\end{array}')
    else:
        lat = lat.replace('\\end{array}','\n\\end{array}')
    lat = lat.replace('\\\\',' \\\\\n')
    lines = lat.split('\n')
    row_widths = (2,) + row_widths
    row_pos = row_widths[0]
    for row_width in row_widths[1:]:
        lines.insert(row_pos, '\n \\addlinespace[0.1cm] \\hline \n \\addlinespace[0.1cm]')
        row_pos = row_pos + row_width + 1
    else:
        lines.insert(row_pos,'\n \\addlinespace[0.1cm] \\hline \n \\addlinespace[0.1cm]')
    headers = [''] * mat.cols
    cursor = 2
    for col_width, col_label in zip(col_widths, col_labels):
        headers[cursor] = sp.latex(col_label)
        cursor = cursor + col_width
    prima_row = ' & '.join(headers)
    prima_row = prima_row + '\\\\'
    lines.insert(1,prima_row)
    return '\n'.join(lines)

if __name__ == '__main__':

    print("Checking that the coupling coefficients diagonalize the tensor product of irrep matrices for the group generators ...")
    total_checks = {}
    full_comparison = {}
    # for group_label in odd_groups:
    for group_label in CPGs.all_group_labels:
        print(group_label, end=' | ')
        all_checks = {}
        full_comparison[group_label] = {}
        group = CPGs.get_group_by_label(group_label)
        irrep_matrices = group.irrep_matrices
        # group_ops = list(group.operations_matrices.keys())
        group_ops = group.generators
        irrep_labels = group.irrep_labels
        for ir0, ir1 in product(irrep_labels, irrep_labels):
            checks = []
            for group_op in group_ops:
                Mir0 = irrep_matrices[ir0][group_op]
                Mir1 = irrep_matrices[ir1][group_op]
                Mir0Mir1 = TensorProduct(Mir0,Mir1)
                CG_sector = group.CG_coefficients_partitioned[(ir0, ir1)]
                # determine which other irs figure in the enclosed coefficients
                listcomp = set(map(lambda x: x[-1],list(CG_sector.keys())))
                ir2s = [[k for k, v in group.component_labels.items() if comp in v][0] for comp in listcomp]
                ir2s = list(set(ir2s))
                # sort them according to their dimension
                ir2s = sorted(ir2s, key=lambda x: group.irrep_dims[x] )
                col_comps = []
                for ir2 in ir2s:
                    col_comps.extend(group.component_labels[ir2])
                row_comps = list(product(group.component_labels[ir0], group.component_labels[ir1]))
                CG_matrix = [[CG_sector[(*row_comp,col_comp)] for col_comp in col_comps] for row_comp in row_comps]
                CG_matrix = sp.Matrix(CG_matrix)
                lhs = sp.simplify(sp.physics.quantum.Dagger(CG_matrix)*Mir0Mir1*CG_matrix)
                rhs = sp.Matrix(sp.BlockDiagMatrix(*[group.irrep_matrices[ir2][group_op] for ir2 in ir2s]))
                check =  (sp.simplify(sp.N(rhs - lhs,chop=True)) == sp.zeros(rhs.rows))
                checks.append(check)
                full_comparison[group_label][(ir0,ir1,group_op)] = sp.simplify(lhs-rhs)
            checks = set(checks)
            if len(checks) == 1 and list(checks)[0]:
                print("✓", ir0, ir1) 
                all_check = True
            else:
                print("X", ir0, ir1)
                all_check = False
            all_checks[(ir0,ir1)] = all_check
        flat_checks = list(set(all_checks.values()))
        if len(flat_checks) == 1 and flat_checks[0]:
            total_checks[group_label] = True
        else:
            total_checks[group_label] = False
        print(total_checks[group_label])

    if all(total_checks.values()):
        print("All coupling coefficients behave as expected.")
    else:
        print("WARNING: some coupling coefficients are misbehaving.")
        print(total_checks)

    print("Producing LaTeX output...")

    latex_matrices = {}
    for group_label in CPGs.all_group_labels:
        flat_labels = dict(sum([list(l.items()) for l in list(new_labels[group_label].values())],[]))
        latex_matrices[group_label] = {}
        group = CPGs.get_group_by_label(group_label)
        irrep_labels = group.irrep_labels
        for ir0, ir1 in product(irrep_labels, irrep_labels):
            CG_sector = group.CG_coefficients_partitioned[(ir0, ir1)]
            # determine which other irs figure in the enclosed coefficients
            listcomp = set(map(lambda x: x[-1],list(CG_sector.keys())))
            ir2s = [[k for k, v in group.component_labels.items() if comp in v][0] for comp in listcomp]
            ir2s = list(set(ir2s))
            # sort them according to their dimension
            ir2s = sorted(ir2s, key=lambda x: group.irrep_dims[x] )
            col_comps = []
            for ir2 in ir2s:
                col_comps.extend(group.component_labels[ir2])
            row_comps = list(product(group.component_labels[ir0], group.component_labels[ir1]))
            CG_matrix = [[CG_sector[(*row_comp,col_comp)] for col_comp in col_comps] for row_comp in row_comps]
            CG_matrix = sp.Matrix(CG_matrix)

            row_comps = [(flat_labels[c1],flat_labels[c2]) for c1,c2 in product(group.component_labels[ir0], group.component_labels[ir1])]
            col_comps = []
            for ir2 in ir2s:
                col_comps.extend([flat_labels[x] for x in group.component_labels[ir2]])
            mout = sp.Matrix(sp.BlockMatrix([[sp.Matrix([ir0,ir1]).T,sp.Matrix(col_comps).T],
                            [sp.Matrix(row_comps),CG_matrix]]))
            col_widths = [group.irrep_dims[ir] for ir in ir2s]
            row_widths = []
            current_comp = row_comps[0][0]
            for idx, row_comp in enumerate(row_comps):
                if current_comp != row_comp[0]:
                    if len(row_widths) ==0:
                        row_widths.append(idx)
                    else:
                        row_widths.append(idx-row_widths[-1])
                current_comp = row_comp[0]
            row_widths = tuple(row_widths)
            mout = format_latex_matrix(mout,col_widths,row_widths,ir2s)
            latex_matrices[group_label][(ir0,ir1)] = mout

    mamoth_printout = []
    numcols = {'C_{1}': 4, 'O': 3}
    for group_label in CPGs.all_group_labels:
        this_matrices = latex_matrices[group_label]
        if group_label == 'C_{1}':
            mamoth_printout.append('\n\\begin{multicols}{%d}' % numcols[group_label])
        if group_label == 'O':
            mamoth_printout.append('\n\\end{multicols}')
            mamoth_printout.append('\\begin{multicols}{%d}' % numcols[group_label])
        mamoth_printout.append('\\section{Group $%s$}\n' % group_label)
        for ir0, ir1 in this_matrices.keys():
            mamoth_printout.append('\\subsection{$%s \\times %s$}\n' % (ir0,ir1))
            mamoth_printout.append('\\begin{equation*}\n%s\n\\end{equation*}\n\n' % this_matrices[(ir0,ir1)])
    mamoth_printout.append('\n\\end{multicols}')

    final_out = '\n'.join(mamoth_printout)
    final_out = final_out.replace("^' ", "^{'} ").replace("^'$","^{'}$")
    print("Saving LaTeX output to %s ..." % outfile)
    open(outfile,'w').write(final_out)
