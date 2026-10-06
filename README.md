# DSA 4060 Personalized Movie Recommender

A content-based movie recommender built for the DSA 4060 practical quiz.

## Student Details

| | |
|---|---|
| **Name** | Justice Chawanda |
| **Student ID** | 670444 |
| **Assigned User ID** | **5** |

The assigned user follows the rule given in the quiz brief: the last two digits of the Student ID
`670444` are `44`, and `44 MOD 40 = 4`, so the assigned User ID is `4 + 1 = 5`. Every profile,
score and recommendation in the notebook is computed for User 5 only.

## Project Objective

This project analyses one assigned user's rating history to work out what kind of films they enjoy,
and then recommends five movies they have not yet rated. Preferences are inferred from the genres of
the movies the user rated positively, and candidates are ranked by how closely their genres resemble
that taste profile. The aim is a recommender whose output can be explained — every recommendation is
traceable to a specific movie the user already liked.

## Datasets

| File | Rows | Fields | Description |
|---|---|---|---|
| `movies.csv` | 36 | `movie_id`, `title`, `genres` | The movie catalogue. `genres` is a pipe-separated list of one or more of 16 genre labels, e.g. `Sci-Fi\|Drama`. |
| `ratings.csv` | 440 | `user_id`, `movie_id`, `rating` | Historical ratings from 40 users, on a half-point scale from 1.5 to 5.0. Each user has rated between 9 and 13 movies. |

Both files are complete: there are no missing values, no duplicate rows, no repeated
`(user_id, movie_id)` pairs, and every `movie_id` in `ratings.csv` exists in the catalogue, so the
two files join cleanly. No cleaning or imputation was needed.

## Method / Approach

1. **Profile the user.** User 5's 9 ratings are joined onto the catalogue to recover titles and
   genres. The pipe-separated genre strings are exploded so each `(movie, genre)` pair is one row,
   which gives a mean rating per genre — the evidence used to describe the user's taste.
2. **Define "liked".** Following the brief, the 6 movies rated **4.0 or higher** are treated as
   positive evidence of preference.
3. **Vectorize genres.** Genres are converted to numerical features with scikit-learn's
   **TF-IDF vectorizer**, using `token_pattern=r"[^|]+"` so that each pipe-separated genre stays a
   single token. This matters: the default word pattern would split `Sci-Fi` into `sci` and `fi`
   and invent a genre that does not exist. TF-IDF also down-weights common tags such as `Drama`
   and rewards the rarer ones that actually distinguish films.
4. **Measure similarity.** **Cosine similarity** is computed across all 36 movies, giving a
   36 × 36 similarity matrix.
5. **Score and filter.** The 9 movies the user has already rated are removed, leaving 27
   candidates. Each candidate is scored by its **mean cosine similarity to all six liked movies**,
   so a film has to match the user's taste as a whole rather than resemble one outlier.
6. **Break ties deterministically.** Because genres are coarse, several candidates score
   identically — three tied at 0.303 for the fifth slot. Ties are resolved by a second, data-driven
   key: **genre affinity**, the mean rating the user has actually given to that candidate's genres,
   with `movie_id` as a final tiebreak so results are fully reproducible.
7. **Explain.** Each recommendation is labelled with the liked movie it most closely resembles,
   the genres they share, and the similarity score.

## How to Run the Notebook

**Requirements:** Python 3, `pandas`, `numpy`, `scikit-learn`, and Jupyter (notebook or the VS Code
Jupyter extension).

```bash
pip install pandas numpy scikit-learn jupyter
jupyter notebook DSA4060_Recommender_670444.ipynb
```

Run the cells **in order from top to bottom** (*Kernel → Restart & Run All*). Later cells depend on
variables defined earlier, so the notebook will not work if cells are run out of sequence. Keep
`movies.csv` and `ratings.csv` in the same folder as the notebook, since they are loaded by relative
path. The notebook in this repository is already fully executed with all outputs visible.

## Recommendation Results

User 5's profile is unambiguous. All six movies they rated 4.0 or higher carry the `Drama` tag, and
their three highest-rated films — *The Social Network* (5.0), *Hidden Figures* (4.5) and
*Remember the Titans* (4.5) — are all biographical or sports dramas. By mean rating, `Biography`
scores 4.50, `Sports` 4.25 and `Drama` 4.14, while the only `Action`, `Thriller`, `Animation` and
`Comedy` films they rated sit far below at 2.00–2.50.

The five recommendations, none of which User 5 has rated:

| Rank | Recommended Movie | Genres | Reason |
|---|---|---|---|
| 1 | The Shawshank Redemption | Drama | Pure `Drama` — the one genre shared by all six liked movies. Mean similarity 0.52, the highest of any candidate. |
| 2 | Creed | Sports\|Drama | Exact genre match (similarity 1.00) for *Remember the Titans* (4.5) and *Ford v Ferrari* (4.0). |
| 3 | Moneyball | Sports\|Drama | Same exact `Sports\|Drama` match as *Creed*; the two are mathematically indistinguishable here. |
| 4 | The Notebook | Romance\|Drama | Perfect genre match (similarity 1.00) for *Titanic*, which the user rated 4.0. |
| 5 | Arrival | Sci-Fi\|Drama | Won the three-way tie at 0.303 on genre affinity. The most speculative pick — see interpretation below. |

**Interpretation.** The list reproduces the user's strongest signal directly: *Creed* and *Moneyball*
are exact matches for the sports dramas they already rated highly, and *The Shawshank Redemption*
ranks first because pure `Drama` sits closest to the genre common to every liked film.
*The Notebook* mirrors *Titanic*. *Arrival* is the weakest of the five and is flagged as such — its
`Drama` half is well supported, but its `Sci-Fi` half rests only on *Interstellar*, which the user
rated 3.0, below the "liked" threshold.

## Limitation and Suggested Improvement

**Limitation.** The recommender knows each movie only by its genre tags, and with just one or two of
16 labels per film that signal is too coarse to separate genuinely different movies. *Creed* and
*Moneyball* score identically (0.517) and three candidates tied for fifth place, which is why an
extra tiebreak rule was needed at all. Being purely content-based, the system can also only return
more of what the user already watches — all five recommendations are dramas, and it can never
surface a film the user would love for reasons genres do not capture, such as director, cast or
tone. The binary 4.0 cutoff adds a further loss: *The Social Network* (5.0) and *Titanic* (4.0)
carry equal weight despite a full point between them.

**Suggested improvement.** Make the recommender **hybrid** by adding collaborative filtering over
the 440 ratings already present in `ratings.csv`. User–user similarity across the 40 users would
identify users whose rating patterns resemble User 5's and surface movies those neighbours rated
highly, recovering the cross-genre discoveries genre similarity cannot reach and breaking ties with
real behavioural evidence instead of a rule. A cheaper refinement in the same direction is to
**weight each liked movie by its rating** when averaging similarities, so a 5.0 film pulls harder
than a 4.0 one, and to enrich the features with director, cast, release year and plot keywords so
that `Drama` stops being one undifferentiated bucket.

## Repository Contents

```
DSA4060-Recommender-670444/
├── README.md                            # This file
├── DSA4060_Recommender_670444.ipynb     # Fully executed notebook (Tasks 1–5)
├── movies.csv                           # Movie catalogue (provided)
└── ratings.csv                          # User ratings (provided)
```
