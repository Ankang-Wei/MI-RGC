## MI-RGC

“MI-RGC：基于相互信息特征增强，用于通过区域图卷积预测载体关联”代码和数据集

珀

华中师范大学数学与统计学院魏安康 (weiankng@mails.ccun.edu.cn) 和蒋兴鹏 (xpjiang@mail.ccnu.edu.cn)。

咖啡

- data/V.zip：噬菌体的余弦相似度。
- data/H.csv：优先级的余弦相似度。
- data/high2-adj.zip 是经过过滤并合并后的噬菌体的信息。
- data/high-1909.zip 是每个样本中噬菌体的状态信息。
- data/VH.zip：噬菌体-宿主关联。如果噬菌体与宿主关联，则其标签为 1。否则，标签为 0。


＃＃＃＃代码

＃＃＃＃工具

在注释宏基因组数据时，我们参考了 kneaddata [ https://github.com/biobakery/kneaddata ]和 kraken2 [ https://github.com/DerrickWood/kraken2 ]进行工具下载和安装。

kneaddata在宿主过程中使用的数据集是human_hg38_refMrna。
只需为此目的使用自定义数据。有关构建数据的详细说明，请参阅[ https://github.com/biobakery/kneaddata ]。

```
kneaddata_database--下载人类基因组 bowtie2
kneaddata -i1 /home/q1.fastq.gz -i2 /home/q2.fastq.gz -o output_dir -t 50 -p 50 -db /home/human_hg38_refMrna
```

在注释过程中，Kraken2 使用标准数据库，该数据库大小为 55GB，包含 RefSeq 古细菌、 细菌、病毒、质粒、人类 1 和 UniVec_Core。用户还可以使用自定义数据库；有关如何构建数据库的说明，请参阅[ https://github.com/DerrickWood/kraken2 ]。数据库可从[ https://benlangmead.github.io/aws-indexes/k2 ]下载。Bracken 和 Kraken2 都使用相同的数据库。

```
kraken2 --db /home/Standard --threads 20 --report flag --report TEST.report --output TEST.output --paired q1.fastq.gz q2.fastq.gz
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
python main.py
```

Users can use their **own data** to train prediction models. 

对于**新宿主/噬菌体**，用户可以从NCBI数据库下载DNA，并使用code/features.py计算从DNA衍生的特征。

**Note:** 

In code/features.py, users need to install the iLearn tool [https://ilearn.erc.monash.edu/ or https://github.com/Superzchen/iLearn] and prepare fasta file, this file is DNA sequences of all phages/hosts. (when you use iLearn to compute the DNA features, you should set the parameters k of Kmer as 3.)

Then users use main.py to predict PHI.


#### Contact

Please feel free to contact us if you need any help.
