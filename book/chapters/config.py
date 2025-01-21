from dataclasses import dataclass
from typing import List, Union

import awkward as ak
import numpy as np
import uproot

from servicex import deliver, dataset as servicex_dataset

@dataclass
class sample:
    "Location of data sample"
    # Shorthand name
    name: str

    # Full Rucio dataset name
    ds: Union[str, List[str]]

    # Codegen
    codegen: str

_samples = {
    "sx_f": sample(
        name="sx_f",
        ds="root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.37622528._000013.pool.root.1",
        codegen="atlasr22"
    ),
    "zee_untyped_r21": sample(
        name="ds_zee_untyped",
        ds="rucio://mc16_13TeV:mc16_13TeV.361106.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zee.deriv.DAOD_PHYS.e3601_e5984_s3126_s3136_r10724_r10726_p4164",
        codegen="atlasr21",
    ),
    "zmumu_r21": sample(
        name="ds_zmuumu",
        ds="rucio://mc16_13TeV:mc16_13TeV.361107.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zmumu.deriv.DAOD_PHYS.e3601_e5984_s3126_s3136_r10724_r10726_p4164",
        codegen="atlasr21",
    ),
    "ztautau_r21": sample(
        name="ds_ztautau",
        ds="rucio://mc16_13TeV:mc16_13TeV.361108.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Ztautau.deriv.DAOD_PHYS.e3601_e5984_s3126_s3136_r10724_r10726_p4355",
        codegen="atlasr21",
    ),
    "jz2_exot15_r21": sample(
        name="ds_jz3_exot15",
        ds="rucio://mc16_13TeV:mc16_13TeV.361022.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2W.deriv.DAOD_EXOT15.e3668_s3126_r9364_r9315_p4696",
        codegen="atlasr21",
    ),
    "bphys_r21": sample(
        name="ds_bphys",
        ds=["root://eosatlas.cern.ch//eos/atlas/user/d/daits/mc16_13TeV/DAOD_BPHY4/mc16.999031.P8BEG_23lo_ggX18p4_Upsilon1Smumu_4mu_3pt2.deriv.DAOD_BPHY4.e8304_a875_r10724_r10726_p3712_pUM999999/DAOD_BPHY4.999031._000001.pool.root.1"],
        codegen="atlasr21",
    ),
    "ttbar_r22": sample(
        name="ds_ttbar",
        ds=[],
        codegen="atlasr21",
    )
}

# sx_f means servicex-frontend documentation dataset, need to find a better name for this, will update after I learn how to find ds names
sx_f = _samples["sx_f"]

def deliver_files(query, s: sample):
    """Sends request for data to servicex backend.
    
    Args:
        query: FuncADLQueryPHYSLITE
         s (sample): The sample to create

    Returns:
        List of files returned from servicex backend
    
    """
    spec = {
        'Sample': [{
            'Name': s.name,
            'Dataset': servicex_dataset.FileList(
                [ s.ds ]
            ),
            'Query': query,
            'Codegen': s.codegen,
        }]
    }
    files = deliver(spec, servicex_name="atlasr22")[s.name]
    assert files is not None, "No files returned from deliver! Internal error"
    return files

def flat_array_from_files(files, branch_name):
    data = []
    for file_path in files:
        file = uproot.open(file_path)
        array = file['atlas_xaod_tree;1'][branch_name].arrays(library="ak")[branch_name]
        data.extend(ak.flatten(array))
    return data

def match_eta_phi(jets, jets_to_match) -> ak.Record:
    """Match `jets_to_match` to the `jets` given. There will always be
    at least one jet found.

    The awkward array needs to have leaves called `pt`, `eta`, and `phi`.

    Args:
        jets (_type_): Source jets
        jets_to_match (_type_): Jets to match to `jets`

    Returns:
        _type_: Matched jets 1:1 in `jets` from `jets_to_match`.
    """

    to_match_pt = jets_to_match.pt
    to_match_eta = jets_to_match.eta
    to_match_phi = jets_to_match.phi
    jet_eta = jets.eta
    jet_phi = jets.phi

    pair_eta = ak.cartesian([jet_eta, to_match_eta], axis=1, nested=True)
    pair_phi = ak.cartesian([jet_phi, to_match_phi], axis=1, nested=True)

    delta_eta = np.abs(pair_eta[:, :, :]["0"] - pair_eta[:, :, :]["1"])
    # TODO: Missing wrap around fro phi
    delta_phi = np.abs(pair_phi[:, :, :]["0"] - pair_phi[:, :, :]["1"])

    delta = delta_eta ** 2 + delta_phi ** 2

    # TODO: remove anything larger that 0.2*0.2
    best_match = ak.argmin(delta, axis=2)

    return ak.Record(
        {
            "eta": to_match_eta[best_match],
            "phi": to_match_phi[best_match],
            "pt": to_match_pt[best_match],
        }
    )