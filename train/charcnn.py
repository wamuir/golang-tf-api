"""charcnn.py: classes that might be used when modeling classes.

William Muir <wamuir@gmail.com>, 2020.  No rights reserved.

# C0 1.0 Universal

_Official translations of this legal tool are available [in other languages](
https://creativecommons.org/publicdomain/zero/1.0/legalcode)._

CREATIVE COMMONS CORPORATION IS NOT A LAW FIRM AND DOES NOT PROVIDE LEGAL
SERVICES. DISTRIBUTION OF THIS DOCUMENT DOES NOT CREATE AN ATTORNEY-CLIENT
RELATIONSHIP. CREATIVE COMMONS PROVIDES THIS INFORMATION ON AN "AS-IS" BASIS.
CREATIVE COMMONS MAKES NO WARRANTIES REGARDING THE USE OF THIS DOCUMENT OR THE
INFORMATION OR WORKS PROVIDED HEREUNDER, AND DISCLAIMS LIABILITY FOR DAMAGES
RESULTING FROM THE USE OF THIS DOCUMENT OR THE INFORMATION OR WORKS PROVIDED
HEREUNDER.

## Statement of Purpose

The laws of most jurisdictions throughout the world automatically confer
exclusive Copyright and Related Rights (defined below) upon the creator and
subsequent owner(s) (each and all, an "owner") of an original work of
authorship and/or a database (each, a "Work").

Certain owners wish to permanently relinquish those rights to a Work
for the purpose of contributing to a commons of creative, cultural and
scientific works ("Commons") that the public can reliably and without
fear of later claims of infringement build upon, modify, incorporate in
other works, reuse and redistribute as freely as possible in any form
whatsoever and for any purposes, including without limitation
commercial purposes. These owners may contribute to the Commons to
promote the ideal of a free culture and the further production of
creative, cultural and scientific works, or to gain reputation or
greater distribution for their Work in part through the use and efforts
of others.

For these and/or other purposes and motivations, and without any
expectation of additional consideration or compensation, the person
associating CC0 with a Work (the "Affirmer"), to the extent that he or
she is an owner of Copyright and Related Rights in the Work,
voluntarily elects to apply CC0 to the Work and publicly distribute the
Work under its terms, with knowledge of his or her Copyright and
Related Rights in the Work and the meaning and intended legal effect of
CC0 on those rights.

1. **Copyright and Related Rights.** A Work made available under CC0
    may be protected by copyright and related or neighboring rights
    ("Copyright and Related Rights"). Copyright and Related Rights
    include, but are not limited to, the following:

    i.   the right to reproduce, adapt, distribute, perform, display,
         communicate, and translate a Work;

    ii.  moral rights retained by the original author(s) and/or
         performer(s);

    iii. publicity and privacy rights pertaining to a person's image or
         likeness depicted in a Work;

    iv.  rights protecting against unfair competition in regards to a
         Work, subject to the limitations in paragraph 4(a), below;

    v.   rights protecting the extraction, dissemination, use and reuse
         of data in a Work;

    vi.  database rights (such as those arising under Directive 96/9/EC
         of the European Parliament and of the Council of 11 March 1996
         on the legal protection of databases, and under any national
         implementation thereof, including any amended or successor
         version of such directive); and

    vii. other similar, equivalent or corresponding rights throughout
         the world based on applicable law or treaty, and any national
         implementations thereof.

2. **Waiver.** To the greatest extent permitted by, but not in
    contravention of, applicable law, Affirmer hereby overtly, fully,
    permanently, irrevocably and unconditionally waives, abandons, and
    surrenders all of Affirmer's Copyright and Related Rights and
    associated claims and causes of action, whether now known or
    unknown (including existing as well as future claims and causes of
    action), in the Work (i) in all territories worldwide, (ii) for the
    maximum duration provided by applicable law or treaty (including
    future time extensions), (iii) in any current or future medium and
    for any number of copies, and (iv) for any purpose whatsoever,
    including without limitation commercial, advertising or promotional
    purposes (the "Waiver"). Affirmer makes the Waiver for the benefit
    of each member of the public at large and to the detriment of
    Affirmer's heirs and successors, fully intending that such Waiver
    shall not be subject to revocation, rescission, cancellation,
    termination, or any other legal or equitable action to disrupt the
    quiet enjoyment of the Work by the public as contemplated by
    Affirmer's express Statement of Purpose.

3. **Public License Fallback.** Should any part of the Waiver for any
    reason be judged legally invalid or ineffective under applicable
    law, then the Waiver shall be preserved to the maximum extent
    permitted taking into account Affirmer's express Statement of
    Purpose. In addition, to the extent the Waiver is so judged
    Affirmer hereby grants to each affected person a royalty-free, non
    transferable, non sublicensable, non exclusive, irrevocable and
    unconditional license to exercise Affirmer's Copyright and Related
    Rights in the Work (i) in all territories worldwide, (ii) for the
    maximum duration provided by applicable law or treaty (including
    future time extensions), (iii) in any current or future medium and
    for any number of copies, and (iv) for any purpose whatsoever,
    including without limitation commercial, advertising or promotional
    purposes (the "License"). The License shall be deemed effective as
    of the date CC0 was applied by Affirmer to the Work. Should any
    part of the License for any reason be judged legally invalid or
    ineffective under applicable law, such partial invalidity or
    ineffectiveness shall not invalidate the remainder of the License,
    and in such case Affirmer hereby affirms that he or she will not
    (i) exercise any of his or her remaining Copyright and Related
    Rights in the Work or (ii) assert any associated claims and causes
    of action with respect to the Work, in either case contrary to
    Affirmer's express Statement of Purpose.

4. **Limitations and Disclaimers.**

    a. No trademark or patent rights held by Affirmer are waived,
       abandoned, surrendered, licensed or otherwise affected by this
       document.

    b. Affirmer offers the Work as-is and makes no representations or
       warranties of any kind concerning the Work, express, implied,
       statutory or otherwise, including without limitation warranties
       of title, merchantability, fitness for a particular purpose, non
       infringement, or the absence of latent or other defects,
       accuracy, or the present or absence of errors, whether or not
       discoverable, all to the greatest extent permissible under
       applicable law.

    c. Affirmer disclaims responsibility for clearing rights of other
       persons that may apply to the Work or any use thereof, including
       without limitation any person's Copyright and Related Rights in
       the Work. Further, Affirmer disclaims responsibility for
       obtaining any necessary consents, permissions or other rights
       required for any use of the Work.

    d. Affirmer understands and acknowledges that Creative Commons is
       not a party to this document and has no duty or obligation with
       respect to this CC0 or use of the Work.
"""

import kerastuner as kt
import numpy as np
import scipy.sparse as sparse
import tensorflow as tf


class HyperDefaulter:
    """ HyperDefaulter is a returner of default for KerasTuner methods """

    def Boolean(self, default, *args, **kwargs):
        """ Boolean is HyperParameters.Boolean """
        _ = (self, args, kwargs)
        return default

    def Choice(self, default, *args, **kwargs):
        """ Choice is HyperParameters.Choice """
        _ = (self, args, kwargs)
        return default

    def Fixed(self, value, *args, **kwargs):
        """ Fixed is HyperParameters.Fixed """
        _ = (self, args, kwargs)
        return value

    def Float(self, default, *args, **kwargs):
        """ Float is HyperParameters.Float """
        _ = (self, args, kwargs)
        return default

    def Int(self, default, *args, **kwargs):
        """ Int is HyperParameters.Int """
        _ = (self, args, kwargs)
        return default


class HyperModel(kt.HyperModel):
    """HyperModel is a model class for building a charCNN, fitting it to data
    (optionally w/ hyperparameter search) and saving the fit model to disk.
    """

    def __init__(self, input_len, chars_len, class_len):
        """Method:  __init__
        ------------------------------
        Construct a HyperModel object.

         input_len:  Length of input / max characters (int)
         chars_len:  Length of character encoding (int)
         class_len:  Length of class list (int)
        ------------------------------
        """

        self.input_len = input_len
        self.chars_len = chars_len
        self.class_len = class_len

        self.model = None

    def build(self, hp):
        """Method:  build
        ------------------------------
        Compile an implementation of Zhang, Zhao and LeCun 2015. See arXiv
        1509.01626. This is not the *best* model for the problem (maybe far
        from it), but it is cheap, easy and fast as is, and can be pruned. A
        hyperparameter search is on learning rate. Note that there should be no
        need to call the build method directly. Also, one-hot encoding of
        classes is required by tf.Keras.losses.CategoricalCrossentropy.

         hp:  a KerasTuner.HyperParameters instance
        ------------------------------
        """

        visb1 = tf.keras.layers.Input(
            shape=(self.input_len,), name="inputLayer", dtype=np.int32
        )
        embd1 = tf.keras.layers.Embedding(self.chars_len + 1, 2 ** 6)(visb1)
        conv1 = tf.keras.layers.Convolution1D(
            filters=2 ** 10,
            kernel_size=7,
            kernel_initializer=tf.random_normal_initializer(0.00, 0.02),
        )(embd1)
        actv1 = tf.keras.layers.ReLU()(conv1)
        pool1 = tf.keras.layers.MaxPooling1D(3)(actv1)
        conv2 = tf.keras.layers.Convolution1D(
            filters=2 ** 10,
            kernel_size=7,
            kernel_initializer=tf.random_normal_initializer(0.00, 0.02),
        )(pool1)
        actv2 = tf.keras.layers.ReLU()(conv2)
        pool2 = tf.keras.layers.MaxPooling1D(3)(actv2)
        conv3 = tf.keras.layers.Convolution1D(
            filters=2 ** 10,
            kernel_size=3,
            kernel_initializer=tf.random_normal_initializer(0.00, 0.02),
        )(pool2)
        actv3 = tf.keras.layers.ReLU()(conv3)
        conv4 = tf.keras.layers.Convolution1D(
            filters=2 ** 10,
            kernel_size=3,
            kernel_initializer=tf.random_normal_initializer(0.00, 0.02),
        )(actv3)
        actv4 = tf.keras.layers.ReLU()(conv4)
        conv5 = tf.keras.layers.Convolution1D(
            filters=2 ** 10,
            kernel_size=3,
            kernel_initializer=tf.random_normal_initializer(0.00, 0.02),
        )(actv4)
        actv5 = tf.keras.layers.ReLU()(conv5)
        conv6 = tf.keras.layers.Convolution1D(
            filters=2 ** 10,
            kernel_size=3,
            kernel_initializer=tf.random_normal_initializer(0.00, 0.02),
        )(actv5)
        actv6 = tf.keras.layers.ReLU()(conv6)
        pool3 = tf.keras.layers.MaxPooling1D(3)(actv6)
        flat1 = tf.keras.layers.Flatten()(pool3)
        dens1 = tf.keras.layers.Dense(2 ** 11, activation="relu")(flat1)
        drop1 = tf.keras.layers.Dropout(0.5)(dens1)
        actv7 = tf.keras.layers.ReLU()(drop1)
        dens2 = tf.keras.layers.Dense(2 ** 11, activation="relu")(actv7)
        drop2 = tf.keras.layers.Dropout(0.5)(dens2)
        actv8 = tf.keras.layers.ReLU()(drop2)
        dens3 = tf.keras.layers.Dense(self.class_len)(actv8)
        soft1 = tf.keras.layers.Softmax(name="outputLayer")(dens3)

        self.model = tf.keras.models.Model(inputs=visb1, outputs=soft1)

        loss = tf.keras.losses.CategoricalCrossentropy()

        optimizer = tf.keras.optimizers.Adam(
            lr=hp.Float(
                name="learning_rate",
                min_value=1.0e-4,
                max_value=5.0e-4,
                sampling="log",
                default=1.5e-4,
            ),
            beta_1=0.900,
            beta_2=0.999,
            amsgrad=False,
        )

        self.model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

        return self.model

    def search(
        self,
        *args,
        max_trials=8,
        executions_per_trial=2,
        directory=None,
        **kwargs
    ):
        """Method:  search
        ------------------------------
        Conduct a hyperparameter search via a Bayesian optimizer. Consider
        supplying a tf.keras.utils.Sequence object, which can retrieve data
        (e.g., from a database) and performing encoding on the fly. See
        tf.Keras.Model.fit for arguments.

         directory:             see KerasTuner.BayesianOptimization
         max_trials:            see KerasTuner.BayesianOptimization
         executions_per_trail:  see KerasTuner.BayesianOptimization
         *args, **kwargs:       passed to tf.Keras.Model.fit
        ------------------------------
        """

        tuner = kt.BayesianOptimization(
            self,
            objective="val_accuracy",
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory=directory,
            project_name="hyper-search",
        )

        tuner.search_space_summary()
        tuner.search(*args, **kwargs)
        tuner.results_summary()

        self.model = tuner.get_best_models(num_models=1)[0]

        return self

    def fit(self, *args, summarize=True, **kwargs):
        """Method:  fit
        ------------------------------
        Train the model using default values for hyperparameters. Consider
        supplying a tf.keras.utils.Sequence object, which can retrieve data
        (e.g., from a database) and performing encoding on the fly. See
        tf.Keras.Model.fit for arguments.

         summarize:        flag to summarize compiled model (bool)
         *args, **kwargs:  all arguments are passed to tf.Keras.Model.fit
        ------------------------------
        """

        self.build(HyperDefaulter())
        if summarize:
            print(self.model.summary())
        self.model.fit(*args, **kwargs)

        return self

    def save(self, filepath, *args, **kwargs):
        """Method:  save
        ------------------------------
        Write the model to disk as a Tensorflow SavedModel. If using the Go API
        then a classes.json file should also be written, with order matching
        the integer encoding. See tf.Keras.Model.save for arguments.

         filepath:         see tf.Keras.Model.save
         *args, **kwargs:  arguments passed to tf.Keras.Model.save
        ------------------------------
        """

        self.model.save(filepath, *args, **kwargs)


class Quantizer:
    """" Quantizer is a character encoder."""

    _DEFAULT_ALPHABET = " !\"#$%&'()*+,-./0123456789:;<=>?@[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    _DEFAULT_INPUT_LENGTH = 250
    _DEFAULT_PADDING = "right"
    _DEFAULT_REVERSE = True
    _DEFAULT_DTYPE = np.dtype("int32")

    def __init__(
        self,
        alphabet=None,
        input_length=None,
        padding=None,
        reverse=None,
        dtype=None,
    ):
        """Method:  __init__
        ------------------------------
        Construct a Quantizer object.

         alphabet:      list of symbols to be encoded (string)
         input_length:  the maximum number of symbols to encode (int)
         reverse:       flag to reverse the characters (bool)
         padding:       "left" or "right" for zero-based padding up to
                        INPUT_LENGTH or None, for no padding (string)
         dtype:         the dtype of the array to return (np.dtype)
        ------------------------------
        """

        self.alphabet = alphabet if alphabet else self._DEFAULT_ALPHABET
        self.input_length = (
            input_length if input_length else self._DEFAULT_INPUT_LENGTH
        )
        self.padding = padding if padding else self._DEFAULT_PADDING
        self.reverse = reverse if reverse else self._DEFAULT_REVERSE
        self.dtype = dtype if dtype else self._DEFAULT_DTYPE

        self.char_map = None
        self.fit()

    def fit(self, raw_documents=None, y=None, **fit_params):
        """Method:  fit
        ------------------------------
        Create an integer mapping for characters.
        The value zero (0) is reserved.

         raw_documents:  will be discarded
         y:              will be discarded
         **fit_params:   will be discarded
        ------------------------------
        """

        _ = (raw_documents, y, fit_params)
        self.char_map = {
            char: idx + 1 for idx, char in enumerate(self.alphabet)
        }
        return self

    def fit_transform(self, raw_documents, y=None, **fit_params):
        """Method:  fit_transform
        ------------------------------
        Integer encode sequences of characters. This method will return a
        sparse representation of the output. Conversion to a dense
        representation can be accomplished by calling .todense on the output.
        Note that the value zero (0) is reserved.

         raw_documents:  iterable over raw text documents
         y:              will be discarded
         **fit_params:   will be discarded

        returns X: sparse (CSR) matrix of shape: (n_samples, input_length)
        ------------------------------
        """

        _ = (y, fit_params)
        self.fit()

        return self.transform(raw_documents)

    def transform(self, raw_documents):
        """Method:  transform
        ------------------------------
        Integer encode sequences of characters. This method will return a
        sparse representation of the output. Conversion to a dense
        representation can be accomplished by calling .todense on the output.
        Note that the value zero (0) is reserved.

         raw_documents: iterable over raw text documents

        returns X: sparse (CSR) matrix of shape: (n_samples, input_length)
        ------------------------------
        """

        mat = sparse.lil_matrix(
            (len(raw_documents), self.input_length), dtype=self.dtype
        )

        for i, document in enumerate(raw_documents):
            mat[
                i,
            ] = self._transform(document)

        return mat.tocsr()

    def _transform(self, string):

        padleft = self.padding == "left"

        s = string[:self.input_length]

        arr = np.zeros(self.input_length, dtype=self.dtype)

        for idx in range(len(arr)):
            stridx = (
                (idx + len(s) - self.input_length) if (padleft) else idx
            )
            if (padleft) and (stridx < 0):
                continue
            if stridx >= len(s):
                break
            char = s[-(stridx + 1)] if (self.reverse) else s[stridx]
            arr[idx] = self.char_map.get(char.lower(), 0)

        return arr
