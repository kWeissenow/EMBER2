import sys,os,json
import tempfile
import numpy as np

from arguments import *
from utils_ros import *
from pyrosetta import *
from pyrosetta.rosetta.protocols.minimization_packing import MinMover

os.environ["OPENBLAS_NUM_THREADS"] = "4"



def logo():
    print('*********************************************************************')
    print('\
*           _        ____                _   _                      *\n\
*          | |_ _ __|  _ \ ___  ___  ___| |_| |_ __ _               *\n\
*          | __| \'__| |_) / _ \/ __|/ _ \ __| __/ _` |              *\n\
*          | |_| |  |  _ < (_) \__ \  __/ |_| || (_| |              *\n\
*           \__|_|  |_| \_\___/|___/\___|\__|\__\__,_|              *')
    print('*                                                                   *')
    print("* J Yang et al, Improved protein structure prediction using         *\n* predicted interresidue orientations, PNAS, 117: 1496-1503 (2020)  *")
    print("* Please email your comments to: yangjy@nankai.edu.cn               *")
    print("* Modified by K. Weissenow for EMBER2 structure predictions.        *")
    print('*********************************************************************')



def main():

    ########################################################
    # process inputs
    ########################################################

    logo()
    # read params
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    with open(scriptdir + '/data/params.json') as jsonfile:
        params = json.load(jsonfile)

    # get command line arguments
    args = get_args(params)
    print(args)

    # init PyRosetta
    init('-hb_cen_soft -relax:default_repeats 5 -default_max_cycles 200 -out:level 100')

    # Create temp folder to store all the restraints
    tmpdir = tempfile.TemporaryDirectory(prefix=args.wdir+'/')
    params['TDIR'] = tmpdir.name
    print('temp folder:     ', tmpdir.name)

    # read and process restraints & sequence
    npz = np.load(args.NPZ)
    seq = read_fasta(args.FASTA)
    L = len(seq)
    params['seq'] = seq
    rst = gen_rst(npz,tmpdir,params)
    seq_polyala = 'A'*len(seq)


    ########################################################
    # Scoring functions and movers
    ########################################################
    sf = ScoreFunction()
    sf.add_weights_from_file(scriptdir + '/data/scorefxn.wts')

    sf1 = ScoreFunction()
    sf1.add_weights_from_file(scriptdir + '/data/scorefxn1.wts')

    sf_vdw = ScoreFunction()
    sf_vdw.add_weights_from_file(scriptdir + '/data/scorefxn_vdw.wts')

    sf_cart = ScoreFunction()
    sf_cart.add_weights_from_file(scriptdir + '/data/scorefxn_cart.wts')

    mmap = MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(False)
    mmap.set_jump(True)

    min_mover = MinMover(mmap, sf, 'lbfgs_armijo_nonmonotone', 0.0001, True)
    min_mover.max_iter(1000)

    min_mover1 = MinMover(mmap, sf1, 'lbfgs_armijo_nonmonotone', 0.0001, True)
    min_mover1.max_iter(1000)

    min_mover_vdw = MinMover(mmap, sf_vdw, 'lbfgs_armijo_nonmonotone', 0.0001, True)
    min_mover_vdw.max_iter(500)

    min_mover_cart = MinMover(mmap, sf_cart, 'lbfgs_armijo_nonmonotone', 0.0001, True)
    min_mover_cart.max_iter(1000)
    min_mover_cart.cartesian(True)

    repeat_mover = RepeatMover(min_mover, 3)


    ########################################################
    # initialize pose
    ########################################################
    pose = pose_from_sequence(seq, 'centroid' )

    # mutate GLY to ALA
    #for i,a in enumerate(seq):
    #    if a == 'G':
    #        mutator = rosetta.protocols.simple_moves.MutateResidue(i+1,'ALA')
    #        mutator.apply(pose)
    #        print('mutation: G%dA'%(i+1))

    set_random_dihedral(pose)
    remove_clash(sf_vdw, min_mover_vdw, pose)

    ########################################################
    # minimization
    ########################################################

    if args.mode == 0:

        # short
        print('short')
        add_rst(pose, rst, 1, 12, params)
        repeat_mover.apply(pose)
        min_mover_cart.apply(pose)
        remove_clash(sf_vdw, min_mover1, pose)

        # medium
        print('medium')
        add_rst(pose, rst, 12, 24, params)
        repeat_mover.apply(pose)
        min_mover_cart.apply(pose)
        remove_clash(sf_vdw, min_mover1, pose)

        # long
        print('long')
        add_rst(pose, rst, 24, len(seq), params)
        repeat_mover.apply(pose)
        min_mover_cart.apply(pose)
        remove_clash(sf_vdw, min_mover1, pose)

    elif args.mode == 1:

        # short + medium
        print('short + medium')
        add_rst(pose, rst, 3, 24, params)
        repeat_mover.apply(pose)
        min_mover_cart.apply(pose)
        remove_clash(sf_vdw, min_mover1, pose)

        # long
        print('long')
        add_rst(pose, rst, 24, len(seq), params)
        repeat_mover.apply(pose)
        min_mover_cart.apply(pose)
        remove_clash(sf_vdw, min_mover1, pose)

    elif args.mode == 2:

        # short + medium + long
        print('short + medium + long')
        add_rst(pose, rst, 1, len(seq), params)
        repeat_mover.apply(pose)
        min_mover_cart.apply(pose)
        remove_clash(sf_vdw, min_mover1, pose)


    # mutate ALA back to GLY
    #for i,a in enumerate(seq):
    #    if a == 'G':
    #        mutator = rosetta.protocols.simple_moves.MutateResidue(i+1,'GLY')
    #        mutator.apply(pose)
    #        print('mutation: A%dG'%(i+1))


    ########################################################
    # full-atom refinement
    ########################################################

    if args.fastrelax == True:

        sf_fa = create_score_function('ref2015')
        sf_fa.set_weight(rosetta.core.scoring.atom_pair_constraint, 5)
        sf_fa.set_weight(rosetta.core.scoring.dihedral_constraint, 1)
        sf_fa.set_weight(rosetta.core.scoring.angle_constraint, 1)

        mmap = MoveMap()
        mmap.set_bb(True)
        mmap.set_chi(True)
        mmap.set_jump(True)

        relax = rosetta.protocols.relax.FastRelax()
        relax.set_scorefxn(sf_fa)
        relax.max_iter(200)
        relax.dualspace(True)
        relax.set_movemap(mmap)

        pose.remove_constraints()
        switch = SwitchResidueTypeSetMover("fa_standard")
        switch.apply(pose)

        print('relax...')
        params['PCUT'] = 0.15
        add_rst(pose, rst, 1, len(seq), params, True)
        relax.apply(pose)

    ########################################################
    # save final model
    ########################################################
    pose.dump_pdb(args.OUT)


if __name__ == '__main__':
    main()

