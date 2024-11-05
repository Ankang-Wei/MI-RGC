## MI-RGC

Code and Datasets for "MI-RGC: A novel framework for predicting phage-host interactions via feature augmentation and regional graph convolution"

#### Developers

Ankang Wei (weiankng@mails.ccun.edu.cn) and Xingpeng Jiang (xpjiang@mail.ccnu.edu.cn) from School of Mathematics and Statistics, Central China Normal University.

#### Datasets

- data/V.zip: Cosine similarity of phage.
- data/H.csv: Cosine similarity of host.
- data/high2-adj.zip is the mutual information of phages after filtering and merging.
- data/high-1909.zip is the state information of phages in each sample.
- data/VH.zip: phage-host associations. If the phage is associated with host, its label will be 1. Otherwise, the label will be 0.
- data/PHI_pairs.csv: The PHIs for the PHI dataset, including the names of the phages, hosts, and their corresponding NCBI accession numbers.
- data/CHERRY_pairs.csv: The PHIs for the CHERRY dataset, including the names of the phages, hosts, and their corresponding NCBI accession numbers.
- data/PHD_pairs.csv: The PHIs for the PHD dataset, including the names of the phages, hosts, and their corresponding NCBI accession numbers.

#### Code

#### Tool

When annotating metagenomic data, users can refer to kneaddata [https://github.com/biobakery/kneaddata] and kraken2 [https://github.com/DerrickWood/kraken2] for tool downloading and installation. 

The dataset used by kneaddata in the host removal process is human_hg38_refMrna. 
You can also use a custom dataset for this purpose. Detailed instructions for constructing the dataset can be found at [https://github.com/biobakery/kneaddata].

```
kneaddata_database --download human_genome bowtie2
kneaddata -i1 /home/q1.fastq.gz -i2 /home/q2.fastq.gz -o output_dir -t 50 -p 50 -db /home/human_hg38_refMrna
```

During the annotation process, Kraken2 uses the Standard database, which is 55GB in size and includes RefSeq archaea, bacteria, viral, plasmid, human1, and UniVec_Core. Users can also use a custom database; for instructions on how to construct one, please refer to [https://github.com/DerrickWood/kraken2]. The database can be downloaded from [https://benlangmead.github.io/aws-indexes/k2]. Both Bracken and Kraken2 use the same database.

```
kraken2 --db /home/Standard  --threads 20 --report flag --report TEST.report --output TEST.output  --paired q1.fastq.gz q2.fastq.gz
bracken -d /home/Standard -i TEST.report -o TEST.S.bracken -w TEST.S.bracken.report -r 150 -l S
```

The package used for computing logical relationships is written in C++ and users can directly call the **high-2.cpp** file for its usage.

##### Environment Requirement

The required packages are as follows:

- Python == 3.8.3
- Keras == 2.8.0
- Tensorflow == 2.3.0
- Numpy == 1.23.5
- Pandas == 1.5.3
- Protobuf == 3.20.3

##### Usage

```
git clone https://github.com/weiankang258369/MI-RGC
cd MI-RGC/code
python main_mi.py
```

##### User

Users can use their **own data** to train prediction models. 

Users first need to download the fasta files for Phage and Host, resulting in the corresponding name directories phage_name.csv and host_name.csv. Then, they should use the **code/feature.py** file to compute the k-mer features for the entities.

```
python feature.py
```
Then, use **code/cos_sim.py** to process the obtained k-mer feature CSV files to generate the similarity matrix for Phages (V.csv) and the similarity matrix for Hosts (H.csv).

```
python cos_sim.py
```
Input phage_name.csv and host_name.csv to use **VH.py** to obtain the association matrix (VH.csv).

```
python VH.py
```
If the user has generated the mutual information file high2-adj.csv using **high-2.cpp**, they can input high2-adj.csv, V.csv, H.csv, and VH.csv into **main.py** to run the script. If high2-adj.csv has not been generated, the user can input V.csv, H.csv, and VH.csv into main.py and delete the section related to high2-adj.csv in the input part of **main.py**. Running this will also yield prediction results, but the embedding process corresponding to this result will no longer include the feature enhancement module.
```
python main.py
```

**Note:** 

If the user wants to predict the hosts of phages but does not have their own known associations, they can use the data/PHI_pairs.csv file to obtain the dataset used in this study. During this process, the target phages should simply be merged into the phage directory in the PHI dataset.


#### Contact

Please feel free to contact us if you need any help.
