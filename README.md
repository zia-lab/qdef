![](./images/qdef-banner.png)
***

`qdef` is a collection of code and data that can be used to calculate the electronic structure of transition metal ions in crystals. For a given crystal host and dopant species, `qdef` can give a picture of the qualitative level structure, and it can also give approximate quantitative results for energy levels.

For an overview of the functions and data included here check out [this Jupyter notebook](https://github.com/zia-lab/qdef/blob/main/qdef.ipynb).

## Tanabe-Sugano Diagrams, Coupling Coefficients, and Others

`qdef` was used to calculate a document with generic and approximate Tanabe-Sugano diagrams for the transition metal ions ([TSK-Diagrams.pdf](https://github.com/zia-lab/qdef/blob/main/docs/TSK-Diagrams.pdf)) using parameters estimated through auxiliary Hartree-Fock calculations (from Fraga). Diagrams are presented under cubic, octahedral, and tetrahedral crystal field strengths. Several Mathematica notebooks (./mathematica) are also included providing an interactive version of this.

![](./images/tsk-example-zn5.png)

`qdef` also includes a document with symbolic expressions for two-electron wave functions under the point group symmetry of the 32 crystallographic point groups, with wave functions represented by determinantal states of spin-orbital symbols. An electronic version of this also included here in the file `./Data/2e-termsm.pkl`.

![](./images/spin-orbitals-and-waves.png)

Here included is also a document ([CFmatrices.pdf](https://github.com/zia-lab/qdef/blob/main/docs/CFmatrices.pdf))  which includes matrix elements for crystal fields in different symmetries.

![](./images/crystal-field-matrices.png)

Finally, a printout ([CGcoefficients.pdf]((https://github.com/zia-lab/qdef/blob/main/docs/CFmatrices.pdf))) of the crystal field coupling coefficients under the crystallographic point groups is also provided.

![](./images/coupling-coeffs-doc.png)


## Acknowledgements

For its calculations `qdef` owes a debt of gratitude to many. 

+ Its group theory tables were obtained from [GTPack](https://gtpack.org).
+ The wonderful collection of coefficients of fractional-parentage from [Velkov](https://www.proquest.com/docview/304605104?) was thoroughly appreciated.
+ The spectroscopic data was parsed from the [spectroscopic databases](https://www.nist.gov/pml/atomic-spectra-database) from NIST.
+ The books of [Griffith](https://www.google.com/books/edition/The_Theory_of_Transition_Metal_Ions/vv08AAAAIAAJ?), [Cowan](https://www.google.com/books/edition/The_Theory_of_Atomic_Structure_and_Spect/avgkDQAAQBAJ?), [Tanabe, Sugano, and Kamimura](https://www.google.com/books/edition/Multiplets_of_Transition_metal_Ions_in_C/ZQHwAAAAMAAJ?) were useful or essenial theoretical references.
+ Hartree-Fock calculations were used  from [Fraga's](https://www.google.com/books/edition/Handbook_of_Atomic_Data/8_fuAAAAIAAJ?) handbook.
+ Conversations with Christopher Dodson and the previous work of Jonathan Kurvits at Brown University were useful and timely.
+ The advisement of Rashid Zia, also at Brown University, was essential.

## Solving the crystal-field Hamiltonian

The accuracy of the calculations offered by `qdef` is restricted by the approximations included in a Hamiltonian that includes the kinetic energy for N valence electrons, a potential with spherical symmetry due to the shielded electric charge of the nucleus, a crystal-field potential that subsumes the electrostatic interaction of valence-electrons with the lattice charges that surround them, the pair-wise Coulomb repulsion between valence electrons, the spin-orbit interaction, and the Trees effective operator. 

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./images/hamiltonian_annotated-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./images/hamiltonian_annotated-light.png">
  <img alt="Single configuration effective Hamiltonian." src="./images/hamiltonian_annotated-dark.png">
</picture>

The subsequent levels of description and analysis for specific ion in given site for a certain crystal host can be separated in the following stages:

+ At the coarsest level of description the valence electrons in the ion are free of all interactions except for the shielded potential of the nucleus. As such what remains is a Hamiltonian of non-interacting electrons, all of which are subject to a potential with spherical symmetry. This being the case, the solution to the multi-electron wave function can be composed of single-electron wave functions, with angular parts equal to spherical harmonics $ \text{Y}_{l,m}$ and radial parts that would come from the solutions to the boundary problem of the radial Schrödinger equation. At this stage each single electron state is labeled by three quantum numbers $n,l,m$, without yet having taken their spin into account.

+ When the crystal-field potential $\text{V}_{CF}$ is brought into the analysis, the overall symmetry of the single electron Hamiltonian is lowered to that of the point group of the site in the crystal. For a given value of the orbital quantum number $l$, the crystal field partially lifts the $(2l+1)$ degeneracy of each single-electron state. This is a straightforward application of group theory, and all that is needed to determine this is the character table of the group. These electrons, whose energies have been split by the crystal field, are then grouped in terms that can be labeled by the irreducible representations of the point group of $\text{V}_{CF}$ , and we shall call them "crystal-free electrons". In general, the crystal-field potential will not mix states with different values of *l*, and as such one can simply use a restricted crystal field potential only defined for the given value of *l*.

+ Then comes the task of forming (or coupling) multi-electron states out of the crystal-free electron states.  At this point it becomes crucial to consider the spin of the composing electrons, which require that the total wave functions should be anti-symmetric on the exchange of any two electrons. One pictures mixing electrons that come from the different irreducible representations into which a crystal field has split a certain value of *l* and to label such states, all of which have the same energy till this point, one uses products of the lower case symbols used for the irreducible representations, these we call "crystal-free electron configurations".

+ Finally one considers the Coulomb repulsion, which will partially lift the degeneracy of the crystal-free electron configurations. The matrix elements of this interaction are diagonal inside of each crystal configuration block, but may also have non-zero contributions outside of these blocks. These contributions are also diagonal in each of the mixed-configuration blocks. If one disregards this off-diagonal terms, then the final energies of each of the terms is simply given as a sum of the energies of the crystal electron configurations and the matrix elements of the Coulomb repulsion. If these elements are not negligible, then the final energies need be found by diagonalizing the matrices that blend together the energies of the considered configurations, and the matrix elements of the Coulomb repulsion between electrons belonging to these configurations.

## Missing Parts

Another price that a host enacts of a dopant is that phonons from the crystal lattice will often participate in the electronic transitions of the dopant, this gives rise to broadening of its transition lines. This is a temperature-dependent phenomenon, of which `qdef` makes no attempt to include in its description.

Another caveat is that the host offers a surrounding electromagnetic medium to the dopant which will vary its optical properties as a function of wavelength. In consequence, another allowance that must be made to better approximate the optical properties on a wide range of wavelengths, is that of providing a sense of the dispersive properties of the crystal host. In `qdef` this is accomplished by integrating into the analysis, models for the refractive index of the considered host as a function of wavelength.

## References

+ *Sugano, Satoru, Yukito Tanabe, and Hiroshi Kamimura*. **Transition-Metal Ions in Crystals**, 1970.
+ *Griffith, John Stanley.* **The Theory of Transition-Metal Ions.** Cambridge University Press, 1964.
+ *Cowan, Robert Duane.* **The Theory of Atomic Structure and Spectra.** Los Alamos Series in Basic and Applied Sciences 3. Berkeley: University of California Press, 1981.
+ *Fraga, Serafin, Jacek Karwowski, and K Saxena.* **Handbook of Atomic Data**. Elsevier Scientific Publishing Company, New York, NY, 1976.


## 

+ This code was written by Juan-David Lizarazo-Ferro during his graduate studies in the Physics Department at Brown University under the advisement of Rashid Zia.
