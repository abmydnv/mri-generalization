# Data access status

Active route: IXI healthy-volunteer brain MRI, openly downloadable under CC BY-SA 3.0, with official age metadata. The project now includes an age-regression pilot; it makes no disease-detection claim. See [ixi-protocol.md](ixi-protocol.md). OASIS and MIRIAD notes below document earlier access investigations.

OASIS-1 was the initial proposed dataset. No OASIS images have been downloaded. The project owner confirmed that they have no current institutional email or research affiliation; the current OASIS application route is therefore unavailable. Do not bypass the application using third-party mirrors or claim an affiliation.

Alternative under investigation: UCL MIRIAD, a longitudinal Alzheimer MRI dataset with registration and manual account approval. The official documentation does not establish that an independent applicant will be approved; access remains unverified. Avoid repeated scans across participant splits. No application has been submitted and no data have been downloaded.

MIRIAD documentation: https://www.ucl.ac.uk/brain-sciences/ion/research/research-centres/dementia-research-centre/research-clinical-trials/minimal-interval-resonance-imaging-alzheimers-disease-miriad

The official access form requires an institutional email and a research statement. The applicant must provide accurate affiliation and agree to the data-use terms. An old university email printed on a publication is not evidence of current access or affiliation.

Official access page: https://sites.wustl.edu/oasisbrains/home/access/

## Draft research statement for applicant review

Aims: evaluate how participant-disjoint partitioning and simple model choices affect measured generalization in cross-sectional brain MRI classification. Reproduce the known bias associated with slice-level partitioning as a clearly labeled invalid comparison.

Methods: define an older-adult cohort after reviewing the data dictionary; exclude repeat reliability sessions from the primary analysis; construct fixed participant-level partitions; compare a majority baseline, demographic baselines, and a small convolutional neural network. Select models using development data only. Report participant-level balanced accuracy, sensitivity, specificity, AUROC, uncertainty, and confounding limitations. Do not use models for clinical decision-making.

Variables of interest: T1-weighted MRI, participant/session identifiers, age, sex, clinical dementia rating, and available diagnostic variables. Final target terminology will follow the documented labels; CDR-positive status will not automatically be described as confirmed Alzheimer's disease.

Only code and permitted aggregate results will be published. Data access is pending; this statement has not been submitted.
