# ToxicCommentClassification
Analysing and desinginh various neural model over the Toxic Comment Classification Dataset provided by Kaggle


## The Problem

  Examining things that a person care about can be troublesome. The risk of maltreatment and badgering on the web implies that numerous individuals quit conveying everything that needs to be conveyed and abandon the media platform. Many social media sites struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.

  The Conversation AI group, an exploration activity established by Jigsaw and Google (both a piece of Alphabet) are chipping away at apparatuses to help improve the online discussion.

  S1One zone of center is the investigation of negative online practices, as dangerous remarks (for example remarks that are impolite, discourteous or generally prone to make somebody leave a talk).

  So far they&#39;ve assembled a scope of openly accessible models served through the Perspective API, including poisonous quality. In any case, the present models still make blunders, and they don&#39;t enable clients to choose which kinds of danger they&#39;re keen on finding (for example a few stages might approve of foulness, however not with different kinds of poisonous substance).

## Methodology

  In this section we study baseline methods for the above mentioned common challenges. Further, we propose our Ensembled Sequential learning architecture. Its goal is to minimize errors by detecting optimal methods for a given comment.

  We applied 8 model design with different accuracy.

  a. LSTM model using Glove word vector
  b. 3 layer deeper LSTM model using Glove word vector
  c. ResNet of depth 3 in LSTM model using Glove word vector
  d. Bidirectional LSTM model using Glove word vector
  e. 3 layer deeper Bidirectional LSTM model using Glove word vector
  f. ResNet of depth 3 in Bidirectional LSTM model using Glove word vector
  g. Bidirectional LSTM model with character encodings
  h. Bidirectional LSTM model using Glove word vector and character encodings

## Data Annotations

**Data Set in use: Wikipedia Talk Pages dataset**

The dataset is under [CC0](https://creativecommons.org/share-your-work/public-domain/cc0/), with the underlying comment text being governed by [Wikipedia&#39;s CC-SA-3.0](https://creativecommons.org/licenses/by-sa/3.0/)

We analyze a dataset published by Google Jigsaw in December 2017 over the course of the &#39;Toxic Comment Classification Challenge&#39; on Kaggle. It includes 223,549 annotated user comments collected from Wikipedia talk pages and is the largest publicly available for the task. These comments were annotated by human raters with the six labels:-

- toxic
- severe\_toxic
- bscene
- threat
- insult
- Identity\_hate

Comments can be associated with multiple classes at once, which frames the task as a multi-label classification problem. Jigsaw has not published official definitions for the six classes. But they do state that they defined a toxic comment as &quot;a rude, disrespectful, or unreasonable comment that is likely to make you leave a discussion&quot;.

## Implementation

  ### a.  **Single layer LSTM model using Glove vector**

  Our LSTM model takes a sequence of words as input. An embedding layer transforms one-hot-encoded words to dense vector representations and a spatial dropout, which randomly masks 10% of the input words, makes the network more robust.

  To process the sequence of word embeddings, we use an LSTM layer with 128 units, followed by a dropout of 10%. Finally, a dense layer with a sigmoid activation makes the prediction for the multi-label classification and a dense layer with softmax activation makes the prediction for the multi-class classification.



  ### b. **Single layer Bidirectional LSTM using Glove vector**

  Bidirectional RNNs can compensate certain errors on long range dependencies. In contrast to the standard LSTM model, the bidirectional LSTM model uses two LSTM layers that process the input sequence in opposite directions. Thereby, the input sequence is processed with correct and reverse order of words. The outputs of these two layers are averaged.



  ### c. **Three Layer Deeper LSTM model**

  Sometimes going deeper will help us finding pattern which are more complex. Here we went by using same LSTM layers three times. Each layer of 100 units and changing the direction of the LTM layers

 
   ### d.  **Three Layer Deeper Bidirectional LSTM model**

   Here we went deeper in the Bi LSTM model and try seeing what models learns from deeper architecture. We stacked 3 BiLSTM layers each of 100 units.



  ### e. **ResNet of depth 3 in LSTM model using Glove word vector**

   The problem with the deeper models are that they lose the information while going deeper, we call it gradient loss. To overcome this property we implemented the ResNet Structure. Here we use 1st layers LSTM result and concatenate with 3rd layer LSTM.


  ### f. **ResNet of depth 3 in Bidirectional LSTM model using Glove word vector**

   We did similar approach as the model 5 in this model, instead of LSTM we used Bi LSTMs.

  ### g. **Bidirectional LSTM model with character encodings**

  The reason we switched the input was a lot of words in the corpora were not present in the Glove Word vector. Many words were misspelled due to a type or genuinely done to abusive the targeted people and not get detected by a basic regular Expression.


  ### h.  **Bidirectional LSTM model using Glove word vector and character encodings**

  Glove vectors store more information about the words and and the character sequence also helps us. So we concatenated both vectors and use them as input. This resulted in best model of the all above models.


## **Experimental Results**

  The dataset features an unbalanced class distribution. 201,081 samples fall under the majority &#39;clear&#39; class matching none of the six categories, whereas 22,468 samples belong to at least one of the other classes. While the &#39;toxic&#39; class includes 9.6% of the samples, only 0.3% are labeled as &#39;threat&#39;, marking the smallest class.

  Comments were collected from the English Wikipedia and are mostly written in English with some outliers, e.g., in Arabic, Chinese or German language. The domain covered is not strictly locatable, due to various article topics being discussed. Still it is possible to apply a simple categorization of comments as follows:

  1. &#39;community-related&#39;: Example: &quot;If you continue to vandalize Wikipedia, you will be blocked from editing.&quot;
  2. &#39;article-related&#39;: Example: &quot;Dark Jedi Miraluka from the MidRim world of Katarr, Visas Marr is the lone surviving member of her species.&quot;
  3. &#39;off-topic&#39;: Example: &quot;== I hate how my life goes today == Just kill me now.&quot;

  Common Challenges We observe these common challenges for Natural Language Processing in the datasets:

  1. Out-of-vocabulary words. A common problem for the task is the occurrence of words that are not present in the training data. These words include slang or misspellings, but also intentionally obfuscated content.
  2. Long-Range Dependencies. The toxicity of a comment often depends on expressions made in early parts of the comment. This is especially problematic for longer comments (\&gt;50 words) where the influence of earlier parts on the result can vanish.
  3. Misspelled -words . We observed that many words are spelled incorrectly. The reason being the data collected may be of a user not using English as their first language. We also observed that many abusive words we intentionally misspelled for the naive regular expression algorithm to fail. Some of them were basic typing typos.

  We went on to try different models which could fit our dataset. We used the LSTM and Bi LSTM layers and also tried the ResNet Structure.

  For input we used Glove 50d vector and also used character sequence as input. The best model we got was for model using single Bidirectional Layer and input as glove vector and character sequence vectors.

  
  ### 1.  **LSTM model using Glove word vector**

    Train on 127656 samples, validate on 31915 samples

    Epoch 1/3

    127656/127656 [==============================] - 503s 4ms/step - loss: 0.0671 - acc: 0.9776 - val\_loss: 0.0490 - val\_acc: 0.9816

    Epoch 2/3

    127656/127656 [==============================] - 500s 4ms/step - loss: 0.0471 - acc: 0.9823 - val\_loss: 0.0472 - val\_acc: 0.9825

    Epoch 3/3

    127656/127656 [==============================] - 491s 4ms/step - loss: 0.0426 - acc: 0.9836 - val\_loss: 0.0463 - val\_acc: 0.9827



  ### 2. **3 layer deeper LSTM model using Glove word vector**

    Train on 127656 samples, validate on 31915 samples

    Epoch 1/3

    127656/127656 [==============================] - 1107s 9ms/step - loss: 0.0775 - acc: 0.9749 - val\_loss: 0.0522 - val\_acc: 0.9811

    Epoch 2/3

    127656/127656 [==============================] - 1044s 8ms/step - loss: 0.0528 - acc: 0.9809 - val\_loss: 0.0522 - val\_acc: 0.9805

    Epoch 3/3

    127656/127656 [==============================] - 1048s 8ms/step - loss: 0.0490 - acc: 0.9820 - val\_loss: 0.0495 - val\_acc: 0.9821



  ### 3.  **ResNet of depth 3 in LSTM model using Glove word vector**

    Train on 127656 samples, validate on 31915 samples

    Epoch 1/3

    127656/127656 [==============================] - 711s 6ms/step - loss: 0.0715 - acc: 0.9763 - val\_loss: 0.0513 - val\_acc: 0.9813

    Epoch 2/3

    127656/127656 [==============================] - 697s 5ms/step - loss: 0.0508 - acc: 0.9814 - val\_loss: 0.0496 - val\_acc: 0.9818

    Epoch 3/3

    127656/127656 [==============================] - 695s 5ms/step - loss: 0.0468 - acc: 0.9824 - val\_loss: 0.0491 - val\_acc: 0.9819



  ### 4. **Bidirectional LSTM model using Glove word vector**

    Train on 143613 samples, validate on 15958 samples

    Epoch 1/2

    143613/143613 [==============================] - 833s 6ms/step - loss: 0.0582 - acc: 0.9798 - val\_loss: 0.0488 - val\_acc: 0.9823

    Epoch 2/2

    143613/143613 [==============================] - 818s 6ms/step - loss: 0.0441 - acc: 0.9834 - val\_loss: 0.0455 - val\_acc: 0.9833



  ### 5.  **3 layer deeper Bidirectional LSTM model using Glove word vector**

    Train on 111699 samples, validate on 47872 samples

    Epoch 1/2

    111699/111699 [==============================] - 4283s 38ms/step - loss: 0.0723 - acc: 0.9760 - val\_loss: 0.0581 - val\_acc: 0.9793

    Epoch 2/2

    111699/111699 [==============================] - 4280s 38ms/step - loss: 0.0508 - acc: 0.9816 - val\_loss: 0.0506 - val\_acc: 0.9818



  ### 6.  **ResNet of depth 3 in Bidirectional LSTM model using Glove word vector**

    Train on 127656 samples, validate on 31915 samples

    Epoch 1/3

    127656/127656 [==============================] - 1615s 13ms/step - loss: 0.0681 - acc: 0.9771 - val\_loss: 0.0503 - val\_acc: 0.9817

    Epoch 2/3

    127656/127656 [==============================] - 1449s 11ms/step - loss: 0.0493 - acc: 0.9819 - val\_loss: 0.0488 - val\_acc: 0.9823

    Epoch 3/3

    127656/127656 [==============================] - 1315s 10ms/step - loss: 0.0452 - acc: 0.9829 - val\_loss: 0.0468 - val\_acc: 0.9827



  ### 7.  **Bidirectional LSTM model with character encodings**

    Train on 111699 samples, validate on 47872 samples

    Epoch 1/3

    111699/111699 [==============================] - 425s 4ms/step - loss: 0.0667 - acc: 0.9777 - val\_loss: 0.0516 - val\_acc: 0.9816

    Epoch 2/3

    111699/111699 [==============================] - 442s 4ms/step - loss: 0.0464 - acc: 0.9829 - val\_loss: 0.0517 - val\_acc: 0.9818

    Epoch 3/3

    111699/111699 [==============================] - 433s 4ms/step - loss: 0.0419 - acc: 0.9841 - val\_loss: 0.0510 - val\_acc: 0.9826



  ### 8.  **Bidirectional LSTM model using Glove word vector and character encodings**

Train on 111699 samples, validate on 47872 samples

Epoch 1/2

111699/111699 [==============================] - 571s 5ms/step - loss: 0.0603 - acc: 0.9790 - val\_loss: 0.0489 - val\_acc: 0.9822

Epoch 2/2

111699/111699 [==============================] - 543s 5ms/step - loss: 0.0435 - acc: 0.9834 - val\_loss: 0.0456 - val\_acc: 0.9829


 
## Discussion

Here are a few ideas to keep in mind when manually optimizing hyperparameters for RNNs:

- Watch out for _overfitting_, which happens when a neural network essentially &quot;memorizes&quot; the training data. Overfitting means you get great performance on training data, but the network&#39;s model is useless for out-of-sample prediction.
- Regularization helps: regularization methods include l1, l2, and dropout among others.
- So have a separate test set on which the network doesn&#39;t train.
- The larger the network, the more powerful, but it&#39;s also easier to overfit. Don&#39;t want to try to learn a million parameters from 10,000 examples – parameters \&gt; examples = trouble.
- More data is almost always better, because it helps fight overfitting.
- Train over multiple epochs (complete passes through the dataset).
- Evaluate test set performance at each epoch to know when to stop (early stopping).
- In general, stacking layers can help.
- For LSTMs, use the softsign (not softmax) activation function over tanh (it&#39;s faster and less prone to saturation (~0 gradients)).
- Updaters: RMSProp, AdaGrad or momentum (Nesterovs) are usually good choices. AdaGrad also decays the learning rate, which can help sometimes.
- Finally, remember data normalization, MSE loss function + identity activation function for regression.

| **Model** | **Number of Epocs** | **Accuracy** |
| --- | --- | --- |
| LSTM model using Glove word vector | 3 | 98.27 |
| 3 layer deeper LSTM model using Glove word vector | 3 | 98.21 |
| ResNet of depth 3 in LSTM model using Glove word vector | 3 | 98.19 |
| Bidirectional LSTM model using Glove word vector | 2 | 98.33 |
| 3 layer deeper Bidirectional LSTM model using Glove word vector | 2 | 98.18 |
| ResNet of depth 3 in Bidirectional LSTM model using Glove word vector | 3 | 98.27 |
| Bidirectional LSTM model with character encodings | 3 | 98.26 |
| Bidirectional LSTM model using Glove word vector and character encodings | 2 | 98.29 |

- ooHere when comparing all the models we observe that to gain better accuracy we don&#39;t need to build a model with greater depth instead a shallow network with Bi-LSTM works better.
- ooHere we also observe that adding character sequence can improve the accuracy.
- ooTo further improve over the accuracy, we needed to clean the data, as many of the data consisted of visual representation of vulgarity.

## Conclusion

In this work we presented multiple approaches for toxic comment classification. We showed that the approaches make different errors and can be combined into an ensemble with improved accuracy measure. The ensemble especially outperforms when there is high variance within the data and on classes with few examples. Some combinations such as shallow learners with deep neural networks are especially effective. Our error analysis on results of the ensemble identified difficult subtasks of toxic comment classification. We find that a large source of errors is the lack of consistent quality of labels. Additionally most of the unsolved challenges occur due to missing training data with highly idiosyncratic or rare vocabulary. Finally, we observed the use of proper character sequence vector and word vectors combination do help over the use of shallow networks.
