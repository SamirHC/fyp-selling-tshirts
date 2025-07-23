# A Machine Learning Pipeline for Automated T-Shirt Graphics Generation

A BEng Joint Maths and Computer Science dissertation project. The report can be read in the `FinalReport.pdf` file.

## Abstract
A machine learning pipeline is explored for the automatic generation of novel T-shirt print
designs. A continuous data pipeline is developed to ingest and clean marketplace listings, followed
by feature extraction for high-level design characteristics. This includes image segmentation
to isolate print graphics, colour palette extraction using CIELAB space K-means clustering,
multi-label classification for colour-related tags, and large language model inference to extract
semantic content from listing titles. These high-level features drive a chain of generative AI
models for the dynamic generation of prompts and design elements. The outputs are compiled
into SVG via an AST internal representation to enable design manipulation. The proposed
system demonstrates the feasibility of generating commercially viable T-shirt graphics at scale
with minimal human intervention, and provides a foundation for an evolutionary algorithm based
framework for aesthetic evaluation and optimisation in the future.

## Installation

Create a virtual environment, navigate to the `fyp-selling-shirts` directory 
and install the required modules:

```
pip install -r requirements.txt
pip install -e .
```
