## DRS-Parsing as BERT Sequence Labeling
This is a semantic parser converting natuarl language text into Discourse Representation Structure (DRS) defined in Parallel Meaning Bank (PMB) (Abzianidze et al. 2017). The parser using a sequence labeling architecture with pretrained language model - BERT. It produces DRSs by predicting and combining three labels, namely *word senses*, *fragment* and *integration labels*, in which the *fragment* is an underspecified lexical meaning representation that is introduced in Evang (2019).

###### Discourse Representation Structure and Model Data Structure

|<a href="https://ibb.co/w6pxMxR"><img src="https://i.ibb.co/nkQ5z5b/datastruc.png" alt="datastruc" border="0"></a>| 
|:-------------------------------------------------------------------------------------------------------------------:| 
| Three labels ( *fragment*, *Word Sense*, *Integration label* ) that predicted by the BERT sequence labeling model for each token in the text.|

###### The model architecture

|<a href="https://ibb.co/Y2PTrVs"><img src="https://i.ibb.co/ZdG1ZbD/model.png" alt="model" border="0"></a> | 
|:-------------------------------------------------------------------------------------------------------------------:| 
| Our base parser consists of BERT and three linear output layers, namely Linear_s,Linear_f,Linear_i. Each produces sense, fragment, and integration label for the tokens in a single text sequence based on their contextual embedding provided by BERT. |

## Installation

It is strongly recommended that you set up a Python virtual environment for this project, using direnv, Conda, or another tool of your choosing. Once done, install the dependencies as follows:
`pip3 install -r requirements.txt`

###### About Google translator Usage

When parsing non-English language data, the parser translate numerial expressions into English first during post-processing e.g., *zwei* -> "two", in order to "digitalize" as required in PMB. The translation is token by token to avoid any word sense disambiguation. We plan to replace it with pure heuristics mechanism.

## Command line

###### 1. Training

`python main.py train -l en -b 12 -t -e 10 -w 0 -lr 0.00015 -fi -s -fm [*model parameter*] -fd [*devset*] -f [*testset*] -ft [*trainset*] -f3 [*2nd Transet*] -f4 [*3rd Transet*]`

in which the *Devset* and *Trainset* are required for the training. Training can be saved and resumed with `-s` and model parameter file `-fm [*model parameter*]`

###### 2. Testing

* Testing can be done in training mode, if trainsets are provided.

  `python main.py train -l en -b 12 -w 0 -fi -fm [*model parameter*] -fd [*devset*] -f [*testset*] -ft [*trainset*] -f3 [*2nd Transet*] -f4 [*3rd Transet*]`

* There is also dedicated testing mode

  a.  with PMB format testing file

      `python main.py train -l en -fi -fm [*model parameter*] -fv [*...\\label_dict.txt*] -fd [*devset*] -f [*testset*]`

      in which *"label_dict.txt"* is a .json file generated during the training that store all the class information 

  b. with plain text file

     `python main.py train -l en -ez -fi -fm [*model parameter*] -fv [*...\\label_dict.txt*] -fd [*textfile*]`

###### 3. *Counter* Evaluation

For proper DRS evaluation with [*Counter*](https://github.com/RikVN/DRS_parsing/tree/a77a8b7a712984df18b937d47004cddc398f4e16)  (Abzianidze et al. 2017), the parser output DRSs can be found in the *"prediction_dev.txt"* and *"prediction_dev.txt"* files for `trainset` and `devset` in the same file dictionary.

## Our Results

|<a href="https://ibb.co/SRSnTK8"><img src="https://i.ibb.co/Qr0NB6R/results.png" alt="results" border="0"></a>| 
|:-------------------------------------------------------------------------------------------------------------------:| 
|: Our model against other DRS parsers on PMB 3.0 evaluation.|

|<a href="https://ibb.co/mJRgMtZ"><img src="https://i.ibb.co/31TQLW6/4lang.png" alt="4lang" border="0"></a>| 
|:-------------------------------------------------------------------------------------------------------------------:| 
|The performance of our best model in parsing non English PMB evaluation sets compared with the state of the art (Noord et al. 2021).|

## References

Lasha Abzianidze, Johannes Bjerva, Kilian Evang, Hessel Haagsma, Rik van Noord, Pierre Ludmann, Duc-Duy Nguyen, Johan Bos (2017): [The Parallel Meaning Bank: Towards a Multilingual Corpus of Translations Annotated with Compositional Meaning Representations](https://arxiv.org/abs/1702.03964). Proceedings of EACL.

Evang, K. (2019). [Transition-based DRS parsing using stack-LSTMs](https://www.aclweb.org/anthology/W19-1202.pdf). In Proceedings of the IWCS Shared Task on Semantic Parsing.

Noord, R., Toral, A., & Bos, J. (2020). Character-level representations improve DRS-based semantic parsing Even in the age of BERT. [arXiv](https://arxiv.org/abs/2011.04308) preprint arXiv:2011.04308.
