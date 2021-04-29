### kwx 0.1.8 (April 29, 2021)

Changes include:

- Support has been added for gensim 3.8.x and 4.x
- Dependencies in requirement and environment files are now condensed
- An alert for users when the corpus size is to small for the number of topics was added
- An import error for pyLDAvis was fixed

### kwx 0.1.7.3 (March 30, 2021)

Changes include:

- Switching over to an src structure
- Removing the lda_bert method because its dependencies were causing breaks
- Code quality is now checked with Codacy
- Extensive code formatting to improve quality and style
- Bug fixes and a more explicit use of exceptions
- More extensive contributing guidelines
- Tests now use random seeds and are thus more robust

### kwx 0.1.5 (March 15, 2021)

Changes include:

- Keyword extraction and selection are now disjointed so that modeling doesn't occur again to get new keywords
- Keyword extraction and cleaning are now fully disjointed processes
- kwargs for sentence-transformers BERT, LDA, and TFIDF can now be passed
- The cleaning process is verbose and uses multiprocessing
- The user has greater control over the cleaning process
- Reformatting of the code to make the process more clear

### kwx 0.1.0 (Feb 17, 2021)

First stable release of kwx

Additions include:

- Full documentation of the package
- Virtual environment files
- Bug fixes
- Extensive testing of all modules with GH Actions and Codecov
- Code of conduct and contribution guidelines

### kwx 0.0.2.2 (Jan 31, 2021)

The minimum viable product of kwx:

- Users are able to extract keywords using the following methods
  - Most frequent words
  - TFIDF words unique to one corpus when compared to others
  - Latent Dirichlet Allocation
  - Bidirectional Encoder Representations from Transformers
  - An autoencoder application of LDA and BERT combined
- Users are able to tell the model to remove certain words to fine tune results
- Support is offered for a universal cleaning process in all major languages
- Visualization techniques to display keywords and topics are included
- Outputs can be cleanly organized in a directory or zip file
- Runtimes for topic number comparisons are estimated using tqdm
