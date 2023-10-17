from xmask.lhc import install_errors_placeholders_hllhc
import numpy as np


def check_madx_lattices(mad):
    assert mad.globals["qxb1"] == mad.globals["qxb2"]
    assert mad.globals["qyb1"] == mad.globals["qyb2"]
    assert mad.globals["qpxb1"] == mad.globals["qpxb2"]
    assert mad.globals["qpyb1"] == mad.globals["qpyb2"]

    assert np.isclose(mad.table.summ.q1, mad.globals["qxb1"], atol=1e-05)
    assert np.isclose(mad.table.summ.q2, mad.globals["qyb1"], atol=1e-05)

    try:
        assert np.isclose(mad.table.summ.dq1, mad.globals["qpxb1"], atol=1e-01)
        assert np.isclose(mad.table.summ.dq2, mad.globals["qpyb1"], atol=1e-01)

        df = mad.table.twiss.dframe()
        for my_ip in [1, 2, 5, 8]:
            assert np.isclose(df.loc[f"ip{my_ip}"].betx, mad.globals[f"betx_IP{my_ip}"], rtol=1e-03)
            assert np.isclose(df.loc[f"ip{my_ip}"].bety, mad.globals[f"bety_IP{my_ip}"], rtol=1e-03)

        assert df["x"].std() < 1e-8
        assert df["y"].std() < 1e-8
    except:
        print("WARNING: Some sanity checks have failed during the madx lattice check")


def check_xsuite_lattices(my_line):
    tw = my_line.twiss(method="6d", matrix_stability_tol=100)
    print(f"--- Now displaying Twiss result at all IPS for line {my_line}---")
    print(tw[:, "ip.*"])
    # print qx and qy
    print(f"--- Now displaying Qx and Qy for line {my_line}---")
    print(tw.qx, tw.qy)


def build_sequence(mad, mylhcbeam, ignore_cycling=False):
    # Select beam
    mad.input(f"mylhcbeam = {mylhcbeam}")

    mad.input(f"""
      ! Get the toolkit
      call,file=
        "acc-models-lhc/toolkit/macro.madx";
      """)

    mad.input(f"""
      ! Build sequence
      option, -echo,-warn,-info;
      if (mylhcbeam==4){{
        call,file="acc-models-lhc/sequences/lhcb4.seq";
      }} else {{
        call,file="acc-models-lhc/sequences/lhc.seq";
      }};
      option, -echo, warn,-info;
      """)

    # Redefine macro for myslice
    my_slice(mad, slice_factor=4)

    # Slice nominal sequence
    mad.input("exec, myslice;")

    mad.input(f"""
    nrj=6800;
    beam,particle=proton,sequence=lhcb1,energy=nrj,npart=1.15E11,sige=4.5e-4;
    beam,particle=proton,sequence=lhcb2,energy=nrj,bv = -1,npart=1.15E11,sige=4.5e-4;
    """)

    if not ignore_cycling:
        mad.input("""
        !Cycling w.r.t. to IP3 (mandatory to find closed orbit in collision in the presence of errors)
        if (mylhcbeam<3){
        seqedit, sequence=lhcb1; flatten; cycle, start=IP3; flatten; endedit;
        };
        seqedit, sequence=lhcb2; flatten; cycle, start=IP3; flatten; endedit;
        """)

    mad.input("""
        ! Set twiss formats for MAD-X parts (macro from opt. toolkit)
        exec, twiss_opt;
        """)


def apply_optics(mad, optics_file):
    mad.call(optics_file)
    # A knob redefinition
    mad.input("on_alice := on_alice_normalized * 7000./nrj;")
    mad.input("on_lhcb := on_lhcb_normalized * 7000./nrj;")


def my_slice(mad, slice_factor=2):
    mad.input(f"slicefactor = {slice_factor};")
    mad.input("""
        myslice: macro = {
        if (MBX.4L2->l>0) {
          select, flag=makethin, clear;
          select, flag=makethin, class=mb, slice=2;
          select, flag=makethin, class=mq, slice=2 * slicefactor;
          select, flag=makethin, class=mqxa,  slice=16 * slicefactor;  !old triplet
          select, flag=makethin, class=mqxb,  slice=16 * slicefactor;  !old triplet
          select, flag=makethin, class=mqxc,  slice=16 * slicefactor;  !new mqxa (q1,q3)
          select, flag=makethin, class=mqxd,  slice=16 * slicefactor;  !new mqxb (q2a,q2b)
          select, flag=makethin, class=mqxfa, slice=16 * slicefactor;  !new (q1,q3 v1.1)
          select, flag=makethin, class=mqxfb, slice=16 * slicefactor;  !new (q2a,q2b v1.1)
          select, flag=makethin, class=mbxa,  slice=4;   !new d1
          select, flag=makethin, class=mbxf,  slice=4;   !new d1 (v1.1)
          select, flag=makethin, class=mbrd,  slice=4;   !new d2 (if needed)
          select, flag=makethin, class=mqyy,  slice=4 * slicefactor;;   !new q4
          select, flag=makethin, class=mqyl,  slice=4 * slicefactor;;   !new q5
          select, flag=makethin, class=mbh,   slice=4;   !11T dipoles
          select, flag=makethin, pattern=mbx\.,    slice=4;
          select, flag=makethin, pattern=mbrb\.,   slice=4;
          select, flag=makethin, pattern=mbrc\.,   slice=4;
          select, flag=makethin, pattern=mbrs\.,   slice=4;
          select, flag=makethin, pattern=mbh\.,    slice=4;
          select, flag=makethin, pattern=mqwa\.,   slice=4 * slicefactor;
          select, flag=makethin, pattern=mqwb\.,   slice=4 * slicefactor;
          select, flag=makethin, pattern=mqy\.,    slice=4 * slicefactor;
          select, flag=makethin, pattern=mqm\.,    slice=4 * slicefactor;
          select, flag=makethin, pattern=mqmc\.,   slice=4 * slicefactor;
          select, flag=makethin, pattern=mqml\.,   slice=4 * slicefactor;
          select, flag=makethin, pattern=mqtlh\.,  slice=2 * slicefactor;
          select, flag=makethin, pattern=mqtli\.,  slice=2 * slicefactor;
          select, flag=makethin, pattern=mqt\.  ,  slice=2 * slicefactor;
          !thin lens
          option rbarc=false; beam;
          use,sequence=lhcb1; makethin,sequence=lhcb1,makedipedge=true,style=teapot;
          use,sequence=lhcb2; makethin,sequence=lhcb2,makedipedge=true,style=teapot;
          option rbarc=true;
        } else {
          print, text="Sequence is already thin";
        };
          is_thin=1;
        };
    """)