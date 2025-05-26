## @evaluation Submodule

The `@evaluation` module is responsible for validating and testing all stress classification models used in the system. It performs three main functions:

1. **Model Validation**  
   Models are evaluated using standard metrics to assess their ability to detect stress levels accurately and consistently across different categories.

2. **Optimised Dataset**  
   The module filters and transforms the raw dataset into a clean, balanced version tailored for both rule-based and transformer-based models.

3. **Model Testing**  
   All models are tested against the optimised dataset under relative  model consistent preprocessing settings. This includes both scripted (VADER) and semantic (embedding-based).