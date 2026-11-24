"""Group members. Edit MEMBERS if your group is only 3 people (this changes the instance!)."""
from oel_instance import stamp

MEMBERS = [
    ("Abeeha Noor", "534659"),
    ("Tooba Akram", "534701"),
    ("Muhammad Saad Bin Mazhar", "508313"),
    ("Capt Abdul Majid Khan", "534908"),
]
CMS_IDS = [cms for _, cms in MEMBERS]
STAMP = stamp(MEMBERS)   # print/plot this on EVERY output
