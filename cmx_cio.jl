using QCBase
using RDM
using FermiCG
using Printf
using Test
using LinearAlgebra
using Profile
using Random
using PyCall
using Arpack
using JLD2

atoms = []

r = 1
a = 1
#push!(atoms,Atom(1,"H", [0, 0*a, 0*r]))
#push!(atoms,Atom(2,"H", [0, 0*a, 1*r]))
#push!(atoms,Atom(3,"H", [0, 1*a, 2*r]))
#push!(atoms,Atom(4,"H", [0, 1*a, 3*r]))
#push!(atoms,Atom(5,"H", [0, 2*a, 4*r]))
#push!(atoms,Atom(6,"H", [0, 2*a, 5*r]))
#push!(atoms,Atom(7,"H", [0, 3*a, 6*r]))
#push!(atoms,Atom(8,"H", [0, 3*a, 7*r]))
#push!(atoms,Atom(9,"H", [0, 4*a, 8*r]))
#push!(atoms,Atom(10,"H",[0, 4*a, 9*r]))
#push!(atoms,Atom(11,"H",[0, 5*a, 10*r]))
#push!(atoms,Atom(12,"H",[0, 5*a, 11*r]))

push!(atoms,Atom(1,"H", [0.0000000000, 0.0000000000, 0.0000000000]))
push!(atoms,Atom(2,"H", [4.2426406900, 0.0000000000, 0.0000000000]))
push!(atoms,Atom(3,"H", [0.0000000000, 4.2426406900, 0.0000000000]))
push!(atoms,Atom(4,"H", [4.2426406900, 4.2426406900, 0.0000000000]))
push!(atoms,Atom(5,"H", [2.1213203400, 2.1213203400, 3.0000000000]))
push!(atoms,Atom(6,"H", [2.1213203400, 2.1213203400, -3.0000000000]))
#push!(atoms,Atom(7,"H", [2.1213203400, 2.1213203400, 0.0000000000]))


clusters    = [(1:4),(5:8),(9:12)]
init_fspace = [(2,2),(2,2),(2,2)]
clusters    = [(1:2),(3:4),(5:6),(7:8),(9:12)]
init_fspace = [(1,1),(1,1),(1,1),(1,1),(2,2)]
clusters    = [(1:4),(5:6)]
init_fspace = [(2,2),(1,1)]


na = 3
nb = 3


basis = "sto-3g"
mol     = Molecule(0,1,atoms,basis)


# get integrals
mf = FermiCG.pyscf_do_scf(mol)
nbas = size(mf.mo_coeff)[1]
ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));


# define clusters
clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
display(clusters)



d1 = RDM1(n_orb(ints))

e_cmf, U, d1  = FermiCG.cmf_oo(ints, clusters, init_fspace, d1,max_iter_oo=40, verbose=0, gconv=1e-6, method="bfgs")


                            
ints = FermiCG.orbital_rotation(ints,U)

max_roots = 100

cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=max_roots,
                                                   init_fspace=init_fspace, rdm1a=d1.a, rdm1b=d1.b, T=Float64)


clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)

cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

ref_fock = FermiCG.FockConfig(init_fspace)

#cmfstate = FermiCG.TPSCIstate(clusters, ref_fock)


cmfstate = FermiCG.TPSCIstate(clusters)
cmfstate = FermiCG.eye!(cmfstate)


sig = FermiCG.open_matvec_thread(cmfstate, cluster_ops, clustered_ham, nbody=1, thresh=1e-9, prescreen=true)

#FermiCG.clip!(sig, thresh=1e-9)


println("Size of cMF Vector = ")
display(size(cmfstate))
display(cmfstate)
println("Size of Sigma = ")
display(size(sig))
display(sig)

# <H> = (<0|H)|0> = <sig|0>
H1 = dot(sig, cmfstate)

# <HH> = (<0|H)(H|0>) = <sig|sig>
H2 = dot(sig, sig)

println("<H> = ",H1)

println("<HH>",H2)

ovlp = dot(cmfstate, cmfstate)
println("<0|0> = ",ovlp)


println("Hello World")
