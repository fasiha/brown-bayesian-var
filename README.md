# Replicating Aaron Brown's value-at-risk methodology described in Wilmott Magazine

Aaron Brown's essay "Forced by the Sternest Circumstances" ([link](https://storage.googleapis.com/wzukusers/user-28782334/documents/595d0085c545cOE8kmtx/Forced%20by%20the%20Sternest%20Circumstances%20200907.pdf), provided on his [website](https://eraider.com/articles)), describes his practical value-at-risk (VaR) approach that hasn't been taught in textbooks.

This notebook is an attempt to replicate two VaR algorithms described therein—the flawed "historical simulation" method and his method, described as a "perfectly simple and obvious way to estimate VaR that works quite well… in one form or another, often deeply disguised, it is at the heart of all successful VaR systems".

**This second method I have failed to fully recreate.** It produces excellent number of breaks but understates the risk, because the average VaR on break days is significantly different than the average overall VaR.

## [Notebook](./Brown%20VaR.ipynb)

Most readers will likely be best served by reading the Notebook that explains, with prose and Python and plots, what I did and the problem I encountered. [Click here](./Brown%20VaR.ipynb) to see that Notebook: GitHub will render it for you.

If you wish to download the Notebook, you may do so by installing [Git](https://git-scm.com/), then opening your command line (Command Prompt, Terminal app, xterm, etc.) and typing the following:
```console
git clone https://github.com/fasiha/brown-bayesian-var.git
```
This will clone the contents of this Git repository to your computer. You will find the Notebook inside the `brown-bayesian-var` directory.