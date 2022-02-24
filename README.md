# Koopman_pruning
Implementation of Koopman operator theory based pruning in the ShrinkBench framework. Based on the results from Redman et al. ``An Operator Theoretic Perspective on Pruning Deep Neural Networks'' (ICLR 2022). 

# Requirements
The ShrinkBench framework (https://github.com/JJGO/shrinkbench) is required. We made use of this because it not allowed us to report comparisons between our methods and existing methods across a number of different compressions and will enable easy sharing of our algorithms. 

While we mostly used ShrinkBench off the shelf, we made a few modifications. In particular, our approach requires the saving of the DNN parameter values during training. To do this, after each batch we flattened the parameters of our model and stored them in a matrix. This was then saved, and becomes a necessary input to koopman_magnitude_pruning.py. To do the flattening, we inserted the following lines in train.py from the ShrinkBench repository after the checkpoint function (approximately line 145). 

def flat_params_as_numpy(self):
    weights = []
    for name, p in self.model.named_parameters():
        if "linear" in name or "conv" in name:
            weights.append(p.view(-1))
    return torch.cat(weights,0).cpu().detach().numpy()
    
Thank you to Nicholas Guttenberg (GoodAI) for showing me this nice trick. 

# Limitations
Currently, for simplicitly, only functions evaluating Koopman magnitude pruning are available. However, to create other Koopman pruning methods, all that is needed to do is compute all the Koopman eigenvalues. The ExactDMD function in koopman_tools.py has a section doing this that is commented out (because it is not optimized, and therefore takes a significant amount of wall-clock time).  

If there is sufficient interest, I can add the functions performing Koopman gradient pruning. 

# Questions, comments, concerns? 
Please do not hesistate to contact me if you have any questions about implementing these Koopman methods in ShrinkBench, and/or implementing other Koopman based pruning strategies. 

Email: wredman@ucsb.edu 
