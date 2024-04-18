# Explaining Attention mechanism in Transformers: from a gender information perspective

* Research course project for master-level course: Theories of Deep Learning at Oxford

## Abstract
Transformer-based models significantly advanced the state-of-the-art in both linguistic
and computer vision tasks, and the working mechanism of its key component,
the self-attention mechanism, lacks a proper in-house explanation. This paper
illustrates the role of attention mechanism from a token level, focusing on gender
information propagation in language modeling tasks. Start by visualizing attention
weight and quantifying attention flow with depth, I then select a class-specified
gradient-based method named Attentive Class Activation Tokens(AttCAT) [1] to
analyze the mechanism of self-attention units via disentangling information flow.
Inspired by another token-level method for obtaining interpretable dense subspaces
named Densray(DR) [2], I compare the two methods in language modeling tasks
and propose an integrated bias visualization method Attentive Densray(AttDR) to
further look into gender information propagation with depth in attention blocks.
Experiments are performed on several templates and datasets, which show high
similarity in AttCAT and DR scores, and clearer explanation by AttDR.

## code specification
### XAI visualization techniques to explore information flow in BERT
1. for attention methods visualization and MLM information flow visualization:
https://github.com/dream3244/Deep-Learning/blob/main/XAI_Transformers_gender/attention_based_method_visualization_example.ipynb
2. for AttCAT calculation:
https://github.com/dream3244/Deep-Learning/blob/main/XAI_Transformers_gender/attcat_MLM_mat_calculation.ipynb
### Densray calculation and visualization
3. for gender data collecting and preprocessing:
https://github.com/dream3244/Deep-Learning/blob/main/XAI_Transformers_gender/prepare_wiki_data_for_DensRay.ipynb
4. for training densray to get rotation matrix:
https://github.com/dream3244/Deep-Learning/blob/main/XAI_Transformers_gender/Training_for_DR_subspace.ipynb
5. for Densray score calculation per layer:
https://github.com/dream3244/Deep-Learning/blob/main/XAI_Transformers_gender/DR_bias_layers_calculate.ipynb
### visualization and insights
6. for heatmap visualization:
https://github.com/dream3244/Deep-Learning/blob/main/XAI_Transformers_gender/heatmapDR.ipynb
7. for similarity calculating:
https://github.com/dream3244/Deep-Learning/blob/main/XAI_Transformers_gender/AttCAT_DR_similarity.ipynb
