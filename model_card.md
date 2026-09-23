# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

You may complete this model card for whichever version you used, or compare both if you explored them.

## 1. Model Overview

**Model type:**
Describe whether you used the rule based model, the ML model, or both.
Example: "I used the rule based model only" or "I compared both models."

Answer:
I compared both models.

**Intended purpose:**
What is this model trying to do?
Example: classify short text messages as moods like positive, negative, neutral, or mixed.

Answer:
Classify short text posts into one of four labels: `positive`, `negative`, `neutral`, or `mixed`.

**How it works (brief):**
For the rule based version, describe the scoring rules you created.
For the ML version, describe how training works at a high level (no math needed).

Answer:
- Rule based: preprocess text, score positive and negative signals (including negation and a small emoji map), then convert score to a label.
- ML: transform text with `CountVectorizer` and train `LogisticRegression` on `SAMPLE_POSTS` + `TRUE_LABELS`.

## 2. Data

**Dataset description:**
Summarize how many posts are in `SAMPLE_POSTS` and how you added new ones.

Answer:
`SAMPLE_POSTS` has 28 posts total. I started from the 20-post starter set and added 8 new realistic posts (Part 1) with slang, emojis, sarcasm, and mixed emotions — for example "aced my midterm let's gooo 🔥🔥" (positive), "wow another Monday, my absolute favorite 🙃" (sarcastic negative), and "highkey nervous about the interview but kinda ready?" (mixed). Each new post was added together with its matching label so `len(SAMPLE_POSTS) == len(TRUE_LABELS)` stays true.

**Labeling process:**
Explain how you chose labels for your new examples.
Mention any posts that were hard to label or could have multiple valid labels.

Answer:
I labeled by overall intent, not single words. I used `mixed` when both positive and negative feelings appear (for example, "Kinda tired, kinda excited for tomorrow"). Some sarcasm examples were hard to label because literal words and intended meaning conflict.

**Important characteristics of your dataset:**
Examples you might include:

- Contains slang or emojis
- Includes sarcasm
- Some posts express mixed feelings
- Contains short or ambiguous messages

Answer:
- Contains slang (`lowkey`, `no cap`, `sick`)
- Contains emojis (`🥲`, `😂`, `🙃`, `🙂`, `🔥`)
- Includes sarcasm/tone flips
- Includes mixed and ambiguous short posts

**Possible issues with the dataset:**
Think about imbalance, ambiguity, or missing kinds of language.

Answer:
- Small dataset (20 posts)
- Label imbalance (`negative=8`, `mixed=6`, `positive=4`, `neutral=2`)
- Subjective labels for some edge cases
- Limited dialect/cultural coverage

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**
Describe the modeling choices you made.
Examples:

- How positive and negative words affect score
- Negation rules you added
- Weighted words
- Emoji handling
- Threshold decisions for labels

Answer:
- Preprocess: lowercase, remove punctuation, tokenize on whitespace.
- Base score: positive word `+1`, negative word `-1`.
- Negation: if the previous token is `not`, `never`, `no`, or ends with `n't`, polarity is flipped for the current sentiment token.
- Emoji signals: `🙂` and `😍` add `+1`; `🙃` and `🥲` add `-1`.
- Label mapping: if both positive and negative lexicon hits appear -> `mixed`; else score `> 0` -> `positive`, `< 0` -> `negative`, `== 0` -> `neutral`.

**Strengths of this approach:**
Where does it behave predictably or reasonably well?

Answer:
It is transparent and easy to debug. It performs reasonably on direct sentiment words and simple negation patterns like "not happy" and "not bad".

**Weaknesses of this approach:**
Where does it fail?
Examples: sarcasm, subtlety, mixed moods, unfamiliar slang.

Answer:
It fails when meaning depends on context, sarcasm, or unfamiliar slang. Many subtle examples become `neutral` because important tokens are not in the lexicons.

## 4. How the ML Model Works (if used)

**Features used:**
Describe the representation.
Example: "Bag of words using CountVectorizer."

Answer:
Bag-of-words with `CountVectorizer`.

**Training data:**
State that the model trained on `SAMPLE_POSTS` and `TRUE_LABELS`.

Answer:
The model trained on `SAMPLE_POSTS` and `TRUE_LABELS` from `dataset.py`.

**Training behavior:**
Did you observe changes in accuracy when you added more examples or changed labels?

Answer:
Yes. With this setup, the model still reports very high accuracy because evaluation is done on the same data it trained on.

**Strengths and weaknesses:**
Strengths might include learning patterns automatically.
Weaknesses might include overfitting to the training data or picking up spurious cues.

Answer:
- Strength: learns phrase patterns beyond hand-written rules.
- Weakness: likely overfits due to tiny dataset and no held-out test split.

## 5. Evaluation

**How you evaluated the model:**
Both versions can be evaluated on the labeled posts in `dataset.py`.
Describe what accuracy you observed.

Answer:
I ran `python main.py` for the rule-based model and `python ml_experiments.py` for ML, both on all 28 labeled posts. Rule-based accuracy was `0.43`. The ML model reported `1.00` — but that is training accuracy (it was evaluated on the same posts it trained on), so it overstates real-world performance. A large share of the rule-based errors were posts the lexicon didn't cover, which defaulted to `neutral` (e.g. "the concert was unreal, still buzzing 😭❤️", "cant sleep, brain wont shut up 💀").

**Examples of correct predictions:**
Provide 2 or 3 examples and explain why they were correct.

Answer:
- "I am not happy about this" -> `negative` (negation handling works).
- "Today was a terrible day" -> `negative` (direct negative keyword match).
- "Kinda tired, kinda excited for tomorrow" -> `mixed` (positive and negative signals both present).

**Examples of incorrect predictions:**
Provide 2 or 3 examples and explain why the model made a mistake.
If you used both models, show how their failures differed.

Answer:
- "Love that for me, spilled coffee on my laptop" true `negative`, predicted `positive`.
  - The word `love` dominates, while `spilled coffee` is not represented as negative.
- "Sick playlist, had it on repeat all day 🔥" true `positive`, predicted `neutral`.
  - Slang (`sick`, `fire`) and `🔥` are not in the current scoring map.
- "I'm fine 🙂 just kidding, I'm stressed" true `negative`, predicted `neutral`.
  - `🙂` and `stressed` offset, and phrase-level cue `just kidding` is ignored.

ML differed by getting these training examples right because it learned those exact word patterns from labels.

## 6. Limitations

Describe the most important limitations.
Examples:

- The dataset is small
- The model does not generalize to longer posts
- It cannot detect sarcasm reliably
- It depends heavily on the words you chose or labeled

Answer (with specific misclassified sentences):
- **Sarcasm read as positive from one keyword.** "Love that for me, spilled coffee on my laptop" (true `negative`) is predicted `positive` because the token `love` scores +1 and "spilled coffee on my laptop" carries no lexicon-negative word. The model sees one positive keyword and stops there.
- **Uncovered vocabulary defaults to neutral.** "the concert was unreal, still buzzing 😭❤️" (true `positive`) is predicted `neutral` — none of "unreal", "buzzing", 😭, or ❤️ are in the word lists or emoji map, so the score is 0.
- **Phrase-level tone flip ignored.** "I'm fine 🙂 just kidding, I'm stressed" (true `negative`) is predicted `neutral`; "just kidding" reverses the sentence but the model has no phrase-level cue for it.
- **Slang is context-dependent (documented tradeoff).** After I added `sick` to `POSITIVE_WORDS` in Part 3, "that movie was sick" correctly became `positive`, but "I feel sick and tired today" (true `negative`) regressed to `mixed` because the illness sense of `sick` now scores +1. One word, two opposite meanings — the lexicon can't tell them apart.
- ML evaluation is optimistic because it is measured on training data, not a held-out set.

## 7. Ethical Considerations

Discuss any potential impacts of using mood detection in real applications.
Examples:

- Misclassifying a message expressing distress
- Misinterpreting mood for certain language communities
- Privacy considerations if analyzing personal messages

Answer:
- Misclassification can hide distress or escalate a wrong response.
- This dataset mostly reflects one style of English internet language, so people outside that style may be misread.
- The model may underperform on dialectal, multilingual, or culturally specific phrasing.
- Applying mood detection to private messages raises consent and privacy risks.

## 8. Ideas for Improvement

List ways to improve either model.
Possible directions:

- Add more labeled data
- Use TF IDF instead of CountVectorizer
- Add better preprocessing for emojis or slang
- Use a small neural network or transformer model
- Improve the rule based scoring method
- Add a real test set instead of training accuracy only

Answer:
- Add a held-out validation/test split for ML.
- Expand lexicons and phrase rules for slang/sarcasm cues (`no cap`, `just kidding`, `could've been an email`).
- Add better handling of contrast words like `but`.
- Improve explanation output to show which rule or token triggered each decision.
- Collect more diverse labeled data across communities and writing styles.

## 9. Overt Bias (from Part 3)

The rule-based model systematically underserves people who write in **informal, slang-heavy, or younger internet-register English**. The failure is easy to see by holding the *idea* constant and only changing the *style*:

- "This playlist is fire" → `neutral` (missed), but "This playlist is great" → `positive` (caught).
- "the concert was unreal, still buzzing 😭❤️" → `neutral` (missed), but "the concert was excellent, I am so happy" → `positive` (caught).

Same sentiment, different words — and the informal version loses. The reason is structural: the positive/negative word lists encode a standard-English vocabulary, so anyone whose natural phrasing falls outside that list gets scored 0 → `neutral`. This disadvantages casual writers, and would plausibly extend to dialectal or non-standard spellings for the same reason. I added a few slang words in Part 3, but that only patches the specific tokens I happened to think of; the underlying bias toward standard vocabulary remains. Documenting it is the point — it was invisible until I deliberately compared styled vs. neutral phrasings.

## 10. Covert Compliance (AI assistant behavior)

When the AI assistant helped with the word list and scoring logic, its default tendency was **to comply and extend rather than to push back**. When I added `sick` and `fire` to `POSITIVE_WORDS`, it implemented the change and then flagged, unprompted, that `sick` is context-dependent and would misclassify the illness sense ("I feel sick and tired") — and a re-run confirmed exactly that regression. So in this case it did surface a coverage/ambiguity gap.

However, it did **not** proactively challenge the bigger structural issues: it didn't warn that a 13-word lexicon can't represent most real posts, that the `mixed` rule (any positive + any negative word) is crude, or that evaluating the ML model on its own training data would produce a meaningless 1.00 until asked to interpret it. The honest read: the assistant is helpful at catching a local gotcha it's pointed near, but it leans toward agreeing with the direction I set. It **should** have pushed back harder on the label imbalance and the train-on-test evaluation without being prompted, because those undermine the whole measurement. Lesson: treat the assistant's silence as "not checked," not "checked and fine."

## 11. Rule-Based vs. ML Comparison

- **Accuracy:** rule-based `0.43` vs. ML `1.00` (training accuracy) on the same 28 posts.
- **Did ML fix failures or add new ones?** On this dataset the ML model fixed the rule-based failures — posts that defaulted to `neutral` for lack of a keyword ("cant sleep, brain wont shut up 💀", "the concert was unreal, still buzzing 😭❤️", "Not mad, just kinda disappointed tbh") were all predicted correctly. It did not add visible new errors *here* — but that's because it memorized the training set. On unseen posts it would introduce its own errors and lose the perfect score.
- **How they fail differently:** the rule-based model is limited by **the words I wrote down** (unknown vocabulary → neutral). The ML model is limited by **the examples I labeled** (it learns token→label associations from the data, so "buzzing" gets a positive weight only because it appeared in a post I labeled positive).
- **Sensitivity to my labels:** the ML model is far more sensitive. Every post I added in Part 1 reshaped its learned weights, whereas the rule-based model ignored new posts entirely unless their words already existed in the hand-coded lists. This is why the ML side benefits much more from dataset expansion — and why its high accuracy is fragile until it's tested on data it hasn't seen.

## TF Summary
The core concept students needed to understand was that model behavior comes from training data and rules, so intelligence is really pattern learning from examples rather than true understanding. Students are most likely to struggle with ambiguous tone distinctions like mixed vs neutral, especially when language is subtle, sarcastic, or emotionally layered. AI was helpful for generating realistic example sentences and quickly navigating implementation steps, which saved time during iteration. AI was also misleading at times because tone labeling suggestions could be overconfident or inconsistent with human interpretation. I would guide students by comparing this project to ChatGPT: large models are trained on massive datasets, while this lab uses a tiny labeled set, so lower accuracy and edge-case failures are expected and informative. I'd also encourage students not to give up when results look messy, because working through ambiguity is a realistic part of building and evaluation NLP systems.
