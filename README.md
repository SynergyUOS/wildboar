# wildboar
wildboar

## Getting Started

### Prerequisites

### Installation

### Run
1. Write a script with biotools. ([how?](#usage))
2. Activate the cloned conda environment.
   ```console
   $ activate arcgispro-py3-clone
   (arcgispro-py3-clone) $
   ```
3. Run your script.
   ```console
   (arcgispro-py3-clone) $ python path/to/your_script.py
   ```


## Input Required

### Specification
||File Format|Coordnate System|Fields|
|-|-|-|-|
|Biotope Map           |Shapefile|Anything but nothing|비오톱, [...]|
|Environmental Layers  |Esri ASCII raster|ITRF_2000_UTM_K|-|
|Keystone Species Table|CSV|ITRF_2000_UTM_K|{Name}, {Longitude}, {Latitude}|
|Commercial Point Table|CSV|GCS_WGS_1984|위도, 경도, [...]|
|Surveypoint Map       |Shapefile|Anything but nothing|국명, 개체수, [...]|
|Foodchain Info Table  |CSV|-|S_Name, Owls_foods, D_Level, Alternatives_S, [...]|

*(All csv files are considered to be encoded using euc-kr.)*

*(Braces can be replaced by another name, if it sticks to same order.)*

*(Values of 비오톱 field must be one of the values in MEDIUM_CATEGORY_CODE field of [this file](biotools/res/biotope_codes.csv).)*

### Where to Use
||H1|H2|H3|H4|H5|H6|F1|F2|F3|F4|F5|F6|
|-|-|-|-|-|-|-|-|-|-|-|-|-|
|Biotope Map           |✔|✔|✔|✔|✔|✔|✔|✔|✔|✔|✔|✔|
|Environmental Layers  | | | |✔| |✔| | | | | |✔|
|Keystone Species Table| | | |✔| |✔| | | | | | |
|Commercial Point Table| | | | |✔| | | | | | | |
|Surveypoint Map       | | | | | | |✔|✔|✔|✔|✔|✔|
|Foodchain Info Table  | | | | | | |✔|✔|✔|✔|✔| |


## Usage

### Basic Use

### With Arguments

### Full Evaluation


## Funding


## Contributors
The following is a list of the researchers who have helped to improve Biotools by constructing ideas and contributing code.
1. Chan Park (박찬)
2. Suryeon Kim (김수련)
3. Jaeyeon choi (최재연)
4. Aida Ayano (아이다 아야노)
5. Hyeonjin Kim (김현진)

## License
The Biotools is distributed under the Spatial science lab in University of Seoul(UOS), a permissive open-source (free software) license.

![](https://lauos.or.kr/wp-content/uploads/2022/02/융합연구실로고.png)
