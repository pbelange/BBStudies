

# Copying all the files required by pymask:

```bash
rsync -rv phbelang@lxplus.cern.ch:"/afs/cern.ch/eng/lhc/optics/runIII/RunIII_dev/2021_V6/PROTON/opticsfile.*" :/afs/cern.ch/eng/lhc/optics/runIII/RunIII_dev/2021_V6/PROTON/README ./acc-models-lhc/optics
```

```bash
rsync -rv phbelang@lxplus.cern.ch:"/afs/cern.ch/eng/tracking-tools/modules/*module*" ./acc-models-lhc/modules
```

```bash
rsync -rv phbelang@lxplus.cern.ch:"/afs/cern.ch/eng/tracking-tools/beambeam_macros/*.*" ./acc-models-lhc/beambeam_macros
```

```bash
rsync -rv phbelang@lxplus.cern.ch:"/afs/cern.ch/eng/tracking-tools/errors/LHC/*.*" ./acc-models-lhc/errors/LHC
rsync -rv phbelang@lxplus.cern.ch:"/afs/cern.ch/eng/tracking-tools/errors/HL-LHC/*.*" ./acc-models-lhc/errors/HL-LHC
```


```bash
rsync -rv phbelang@lxplus.cern.ch:"/afs/cern.ch/eng/tracking-tools/tools/*.*" ./acc-models-lhc/tools
```


```bash
rsync -rv phbelang@lxplus.cern.ch:/afs/cern.ch/eng/lhc/optics/runIII/RunIII_dev/IR7-Run3seqedit.madx ./acc-models-lhc/
```

```bash
rsync -rv phbelang@lxplus.cern.ch:/afs/cern.ch/eng/lhc/optics/runII/2018/toolkit/macro.madx :/afs/cern.ch/eng/lhc/optics/runII/2018/toolkit/myslice.madx ./acc-models-lhc/
```



# Additionnally, one could dowload the official distribution:
```bash
# Adding modules,tools,beambeam,errors:
git clone https://github.com/lhcopt/lhcmask.git ./acc-models-lhc/modules
git clone https://github.com/lhcopt/lhctoolkit.git ./acc-models-lhc/tools
git clone https://github.com/lhcopt/beambeam_macros.git ./acc-models-lhc/beambeam_macros
git clone https://github.com/lhcopt/lhcerrors.git ./acc-models-lhc/errors
```


Downloaded sequences from  https://gitlab.cern.ch/acc-models/acc-models-lhc/-/tree/2022/
```bash
wget RawPermalink
```

Date: Sept. 27, 2022, permalinks:
https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/2023/lhc.seq
https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/2023/lhcb4.seq

# Check if RFCAVITY PROPERLY INSTALLED:
```bash
sed -n 771p lhc.seq
sed -n 771p lhcb4.seq
```
If not, you can add the harmonic numbers with:
```bash
sed -i "s|ACSCA : RFCAVITY, L := l.ACSCA;|ACSCA : RFCAVITY, L := l.ACSCA, HARMON := HRF400;|" acc-models-lhc/lhc.seq
sed -i "s|ACSCA : RFCAVITY, L := l.ACSCA;|ACSCA : RFCAVITY, L := l.ACSCA, HARMON := HRF400;|" acc-models-lhc/lhcb4.seq
```