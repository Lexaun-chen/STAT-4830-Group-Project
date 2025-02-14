# 1. Problem Statement 

## 1.1 Overview
This week, we incorporated item features into our regression model to better specify the utility values of different items. By doing so, we aim to enhance the explanatory power of our model and improve its ability to predict the likelihood of people purchasing items not included in their previous offer set.
Additionally, we utilized a real-world dataset from Expedia, dividing it into training and testing subsets for model evaluation. To analyze the problem, we applied both linear regression and a neural network framework, comparing their performance and insights.

## 1.2 Background
Expedia offers thousands of hotels, each characterized by attributes such as price, number of rooms, and available space. However, not all hotels are presented to every consumer; we only observe a consumer's choice within their personalized offer set. Our objective is to model consumer decision-making over the broader universal set of options and to predict the likelihood of a consumer selecting a hotel that was not included in their initial offer set.

To address this, we draw upon the concept of utility maximization, a cornerstone of economic theory. Assuming rational behavior, we posit that consumers evaluate hotels based on their utility, which is determined by the features of each option. By estimating these utility values, we aim to build a model that captures consumer choice behavior more accurately and supports the development of enhanced recommendation systems, thereby improving the match between consumer preferences and offered options.

## 1.3 Data

# 2. Technical Approach 

## 2.1 Mathematical Formulation
The fuction
$$
\sigma(w)_i = \frac{e^{wx_i}}{\sum_{j=1}^{K} e^{wx_j}}
$$ 

represents the probability that a consumer selects hotel 𝑖 from a choice set of 
𝐾 hotels, based on a multinomial logit model.
Here, $𝑥_𝑖$ is a feature vector representing the attributes of hotel 𝑖 (e.g., price, number of rooms, location), and 𝑤 is the weight vector capturing the importance of each feature. The term $𝑤𝑥_𝑖$ represents the utility of hotel 
𝑖, as perceived by the consumer.

By incorporating item features into the multinomial logit framework, we compute the probability of each hotel being selected. This approach allows us to model consumer choice behavior more effectively and derive insights into the relative importance of different hotel attributes.

## 2.2 Neural Network 
Next, we consider that the weighted sum of features may fail to capture nonlinear relationships between features and utility values. To address this limitation, we use a neural network to gain deeper insights.

The Deep_MNL neural network calculates utility values through the following structure:

### Input Layer:
Accepts a feature vector of a hotel (𝑖𝑛𝑝𝑢𝑡_𝑑𝑖𝑚) that represents attributes such as price, location, and size.

### Hidden Layers:
Layer 1: A fully connected layer with 100 neurons that applies a linear transformation to the input features. Nonlinearity is introduced using a sigmoid activation function.

Layer 2: Another fully connected layer with 100 neurons, processing the output of Layer 1 with another linear transformation and sigmoid activation.

### Output Layer:

Layer 3: A fully connected layer with a single neuron, which outputs the predicted utility value for the hotel.

### Forward Propagation:

Data flows sequentially through layer1, layer2, and layer3, with sigmoid activations applied after the first two layers.

By replacing the traditional weighted sum with this neural network structure, we can better capture the intricate relationships between hotel features and their utility values, providing a more flexible and accurate framework for modeling consumer choice behavior.

## 2.3 Validation

# 3. Initial Results 

## 3.1 Performance Matrix



# 4. Next Steps 

## 4.1 Future Direction
In future work, we plan to incorporate matrix completion into our approach. Given the high dimensionality of hotel features, running a multinomial logit (MNL) model with all features can be computationally expensive. To address this, we propose leveraging unsupervised learning techniques to efficiently classify both hotels and consumers into distinct clusters. By grouping similar entities, we can reduce the complexity of the problem while preserving meaningful patterns in consumer choice behavior. Matrix completion can then be applied within these clustered datasets to infer missing preferences and improve recommendation accuracy, making the model both scalable and robust.

## 4.2 Detailed Pathways

## 4.3 Key Techinical Challenges
