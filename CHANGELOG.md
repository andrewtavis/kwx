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
