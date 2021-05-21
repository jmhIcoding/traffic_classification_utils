# FS-Net

Implementation of "FS-Net: A Flow Sequence Network For Encrypted Traffic Classification".

If you find this method helpful for your research, please cite this paper:

```latex
@inproceedings{LiuHXCL19,
  author    = {Chang Liu and
               Longtao He and
               Gang Xiong and
               Zigang Cao and
               Zhen Li},
  title     = {FS-Net: {A} Flow Sequence Network For Encrypted Traffic Classification},
  booktitle = {{IEEE} Conference on Computer Communications (INFOCOM), 2019},
  pages     = {1171--1179},
  year      = {2019}
}
```

------

### Requirement

- python >= 3.4
- numpy == 1.14.5
- tqdm
- tensorflow == 1.8.0

------

### Dataset Format

The dataset consists of multiple files, and each file contains all the flow records of a specific application. And the files are ended with `.num`. For example

```
origin_data
	|---- alicdn.num
	|---- baidu.num
```

For a specific application, each flow record is consists with two parts, for example

```
50	3	7	5	5	5	;2920	167	51	78	968	38	
```

There are two sequences in a record: the first one is encoded status sequence and the second on is the packet length sequence. The two sequences are separated with `;`, and the elements in the sequences are separated with `\t`. 

### How to use

#### Step 1. Pre-Process The Dataset

The dataset is first formalized into `.json` files, and the train set and development set are split as follows:

```bash
python main.py --mode=prepro
```

The dataset will saved in the `record` folder, and the files are start with `train` and `test`. The setting can be changed with `--train_json`, `--test_json`, `--train_meta` and `--test_meta`.

#### Step 2: Train The Model

We can train our model by:

```bash
python main.py --mode=train
```

**Note**: hyper-parameters (such as batch size, hidden size, layer number) of the model and the training process can be explored in the `main.py`.

#### Step 3: Evaluation.

Given the evaluation dataset, we can conduct the evaluation with:

```bash
python main.py --mode=test --test_json=xxxxxx --test_model_dir=yyyyy
```

The model will loaded from the `${test_model_dir}`, and the `${test_json}` is the test data. The test data have the same format with the results of the Step 1.
