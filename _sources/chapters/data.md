(ch-data-samples)=

# Data Samples

ATLAS data comes in many flavors. Because the complete data is so large, ATLAS makes a number of different samples that are slimmed down in various ways. The consequences of this can be far reaching: if a derivation removes attributes necessary to perform a calibration for muons, then accessing calibrated muons will not be possible downstream.

In general, we'll be accessing DAOD_PHYS datasets (more information about them can be found [here](https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/DerivationProductionTeam)). These are complete, with all objects, though they have some rather severe object cuts that makes them useless for a number of physics final states (for example, no tracks less than $p_T$ 10 GeV are included).

## R21 (Run 2)

For Run 2, software release R21, we use the following datasets:

| Name         | Events | Rucio Dataset Name     |
|--------------|-|-------------|
| `ds_Zee`     | 69,734,200 | mc16_13TeV.361106.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zee.deriv.DAOD_PHYS.e3601_e5984_s3126_s3136_r10724_r10726_p4164 |
| `ds_Zmumu`   | 292,915,100 | mc16_13TeV.361107.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zmumu.deriv.DAOD_PHYS.e3601_e5984_s3126_s3136_r10724_r10726_p4164 |
| `ds_Ztautau` | | mc16_13TeV.361108.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Ztautau.deriv.DAOD_PHYS.e3601_e5984_s3126_s3136_r10724_r10726_p4164 |
| `ds_bphys` | | mc16.999031.P8BEG_23lo_ggX18p4_Upsilon1Smumu_4mu_3pt2.deriv.DAOD_BPHY4.e8304_a875_r10724_r10726_p3712_pUM999999 |
| `ds_jz2_exot15` | 15989500 | mc16_13TeV.361022.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2W.deriv.DAOD_EXOT15.e3668_s3126_r9364_r9315_p4696 |

You can find the datasets like `ds_Zee` defined in the `config.py` file. If running against a `servicex` backend for R21 files, they are defined as:

```python
from func_adl_servicex_xaodr21 import SXDSAtlasxAODR21PHYS

ds = SXDSAtlasxAODR21PHYS(rucio_dataset_name, backend="xaod")
```

and similarly for Run 3 data:

```python
from func_adl_servicex_xaodr24 import SXDSAtlasxAODR21PHYS, SXDSAtlasxAODR21PHYSLITE

ds_phys = SXDSAtlasxAODR21PHYS(rucio_dataset_name, backend="xaod")
ds_physlite = SXDSAtlasxAODR21PHYSLITE(rucio_dataset_name, backend="xaod")
```

And if running on a local file, then the following definition is made:

```python
from func_adl_servicex import SXLocalxAOD
from func_adl_servicex_xaodr21.event_collection import Event

class xAODLocalTyped(SXLocalxAOD[Event]):
    def __init__(self, file_path: Path):
        super().__init__(file_path, item_type=Event)

ds = xAODLocalTyped(local_path_of_dataset_file)
```

If running on Run 3 data, replace the `xaodr21` part of the package name with `xaodr24` or `xaodr22`.

Note the use of the class `Event`: this defines the types and metadata for the `xAOD` backend and allows you to access all the various methods and types without having to resort to heuristics (which work for simple cases, but not complex ones).

The `ds_jz2_exot15` is a special case - that isn't a `DAOD_PHYS` file. It is used to demonstrate some access patterns that are only possible in derivations. `DAOD_PHYS` has had some of the information removed to keep its size under control. Its declaration is as follows:

```python
from func_adl_servicex_xaodr21 import calib_tools

ds_jz2_exot15 = calib_tools.query_update(
    base_jz2_exot15,
    jet_collection="AntiKt4EMTopoJets",
    jet_calib_truth_collection="AntiKt4TruthJets",
    perform_overlap_removal=False,
)
```

Here, `base_jz2_exot15` is the dataset defined in the normal way above (as running on a local file or a servicex backend). for more details about using `calib_tools` see the [calibration](ch-calibration) chapter.

The datasets were picked on advice from experts in ATLAS and then using the [AMI dataset finder](https://ami.in2p3.fr/), which is a very useful way to scan for datasets you already know must exist.
