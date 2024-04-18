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

