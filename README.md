# Offensive and Slang Word Detection in Romanised Bengali and English Text

Classifies short social media words and phrases as offensive/slang or ordinary. Started as a team course project
(CSE332, Daffodil International University); this version rebuilds the evaluation to measure performance on words the model
has not seen.

**Contributions:** code and analysis by Tamanna Akter Swarna; dataset collected manually from social media comments by the team,
with data collection support from Mayesha Iqbal; original report written by Tasmima Akter.

> The dataset contains offensive language.

## Data
- `data/OffensiveOrSlangWords.csv`: 3,880 rows, columns `Words` and `label` (1 = offensive/slang, 0 = ordinary)
- Only **990 unique** words and phrases (507 ordinary, 483 offensive/slang); most appear exactly four times
- English and romanised Bengali (Bengali written with English letters); 58.9% of unique entries are single words

## Key finding: duplicates inflate the original score
A random 70/30 split of the 3,880 rows gives about 98% accuracy (mean 97.9% over 20 splits), but on average **94.9% of test rows
are exact copies of training rows**. The score reflects memorisation. After de-duplication, evaluation uses data where no test string
appears in training.

## Approach
1. Normalised text (lower-case, whitespace) and removed duplicates (3,880 to 990 unique examples)
2. Reproduced the original random-split protocol over 20 seeds to measure the leakage
3. Compared five models with 5-fold stratified cross-validation (vectorisers fitted inside each fold)
4. Evaluated the chosen model on a 20% held-out set with a confusion matrix
5. Inspected the strongest features and analysed errors

## Results (5-fold cross-validation on unique examples)
| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Word-level logistic regression (baseline) | 0.645 | 0.462 | 0.687 |
| Char n-gram logistic regression | 0.800 | 0.801 | 0.884 |
| Char n-gram linear SVM | 0.810 | 0.810 | 0.891 |
| Char n-gram naive Bayes | 0.779 | 0.790 | 0.863 |
| Char n-gram random forest | 0.756 | 0.752 | 0.848 |

- Character n-gram features lift accuracy from 64.5% to about 80%. Logistic regression and the SVM are statistically indistinguishable
  (fold-to-fold variation about 0.02); logistic regression is chosen for its probabilities and inspectable features.
- Held-out check: 77.3% accuracy on 198 unseen words and phrases.

![Model comparison](images/02_model_comparison.png)
![Confusion matrix](images/03_confusion_matrix.png)

## What the model learned
The strongest features are short letter fragments, so the model partly learns spelling style (romanised Bengali versus English).
Errors cluster in English insults, context-dependent phrases and neutral Bengali, Hindi or Urdu words.

## Limitations
- Small dataset (990 unique examples); labelling process and annotator count not documented
- Single words and short phrases, no sentence context
- Binary labels for a subjective task, and a strong correlation between language and label
- Performance on real comments or other platforms is unknown

## How to run
1. Place `OffensiveOrSlangWords.csv` in a `data/` folder next to the notebook (or in the same folder).
2. Open `slang_word_detection.ipynb` in Google Colab or Jupyter and run all cells.

## Next steps
- Add more varied data, including offensive English words and neutral romanised Bengali words
- Document labelling guidelines and measure annotator agreement
- Move to full comments and test context-aware multilingual models
- Evaluate on a dataset from a different source

## Project history
This project began as a team course project in 2023 (CSE332, Daffodil International University). The original PySpark version
and its report are kept unchanged in `notebooks/original_2023/`. In 2026, with more experience in model evaluation, I found that the
dataset contains only 990 unique strings repeated across 3,880 rows, so a random split placed the same words in both the training and
test sets and inflated the reported accuracy (about 98%). This version corrects the evaluation, and the corrected results are reported above.
