# KArSL Milestone 1 research note

The current official KArSL dataset page is maintained by Hamzah Luqman at
<https://hamzah-luqman.github.io/KArSL/>. It describes KArSL-502 as 502
isolated word-level Arabic Sign Language signs, with RGB, depth, and skeleton
modalities, three professional signers, and 50 repetitions per sign. The
download page currently exposes public Google Drive archives by signer and
train/test split. The official label workbook is `KARSL-502_Labels.xlsx` in
the KArSL-502 root folder.

For this pilot, the manifest fixes the official `test` split, sign IDs
0171--0176, signer IDs 01--03, and the lexicographically first valid RGB MP4
member for every sign/signer pair. Because the official source is distributed
as solid 7z range archives, the acquisition tool downloads only the bounded
prefixes needed for those exact members and extracts no other local videos.

SSHI access at <https://sshi.sa/> has been approved for this project as stated
in the task brief. SSHI scraping/data collection is intentionally deferred to
a later dedicated task; no SSHI scraper or SSHI data is included here.

Primary research citation:

> Sidig, Ala Addin I., Hamzah Luqman, Sabri Mahmoud, and Mohamed Mohandes.
> “KArSL: Arabic Sign Language Database.” *ACM Transactions on Asian and
> Low-Resource Language Information Processing*, 20(1), 2021.

The official page provides the requested citation and links to the paper.
