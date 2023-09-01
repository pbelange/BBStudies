

# Copying all the files required by pymask:

```bash
rsync -rv phbelang@lxplus.cern.ch:/afs/cern.ch/eng/lhc/optics/runIII/RunIII_dev/2021_V6/PROTON/opticsfile.* :/afs/cern.ch/eng/lhc/optics/runIII/RunIII_dev/2021_V6/PROTON/README ./Tracking_tools/Optics
```

```bash
rsync -rv phbelang@lxplus.cern.ch:/afs/cern.ch/eng/tracking-tools/modules/*module* ./Tracking_tools/Modules
```

```bash
rsync -rv phbelang@lxplus.cern.ch:/afs/cern.ch/eng/tracking-tools/beambeam_macros/*.* ./Tracking_tools/Beambeam_macros
```

```bash
rsync -rv phbelang@lxplus.cern.ch:/afs/cern.ch/eng/tracking-tools/errors/LHC/*.* ./Tracking_tools/Errors/LHC
rsync -rv phbelang@lxplus.cern.ch:/afs/cern.ch/eng/tracking-tools/errors/HL-LHC/*.* ./Tracking_tools/Errors/HL-LHC
```


```bash
rsync -rv phbelang@lxplus.cern.ch:/afs/cern.ch/eng/tracking-tools/tools/*.* ./Tracking_tools/Tools
```


```bash
rsync -rv phbelang@lxplus.cern.ch:/afs/cern.ch/eng/lhc/optics/runIII/RunIII_dev/IR7-Run3seqedit.madx ./Tracking_tools/
```

```bash
rsync -rv phbelang@lxplus.cern.ch:/afs/cern.ch/eng/lhc/optics/runII/2018/toolkit/macro.madx :/afs/cern.ch/eng/lhc/optics/runII/2018/toolkit/myslice.madx ./Tracking_tools/
```



# Additionnally, one could dowload the official distribution:
```bash
# Adding modules,tools,beambeam,errors:
git clone https://github.com/lhcopt/lhcmask.git ./Tracking_tools/modules
git clone https://github.com/lhcopt/lhctoolkit.git ./Tracking_tools/tools
git clone https://github.com/lhcopt/beambeam_macros.git ./Tracking_tools/beambeam_macros
git clone https://github.com/lhcopt/lhcerrors.git ./Tracking_tools/errors
```