# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import autograd.numpy as anp

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants \
    import INITIAL_NOISE_VARIANCE, NOISE_VARIANCE_LOWER_BOUND, \
    NOISE_VARIANCE_UPPER_BOUND, DEFAULT_ENCODING, INITIAL_COVARIANCE_SCALE, \
    COVARIANCE_SCALE_LOWER_BOUND, COVARIANCE_SCALE_UPPER_BOUND
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.distribution \
    import Gamma, LogNormal
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon \
    import Block
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon_blocks_helpers \
    import encode_unwrap_parameter, register_parameter, create_encoding
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel \
    import KernelFunction
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean \
    import ScalarMeanFunction, MeanFunction
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_state \
    import GaussProcPosteriorState


class MarginalLikelihood(Block):
    """
    Marginal likelihood of Gaussian process with Gaussian likelihood

    :param kernel: Kernel function (for instance, a Matern52---note we cannot
        provide Matern52() as default argument since we need to provide with
        the dimension of points in X)
    :param mean: Mean function which depends on the input X only (by default,
        a scalar fitted while optimizing the likelihood)
    :param initial_noise_variance: A scalar to initialize the value of the
        residual noise variance
    :param has_covariance_scale: If True, we maintain a `covariance_scale`
        parameter here, which is multiplied with the kernel. This is useful
        if several likelihoods share the same kernel, but have their own
        covariance scales
    """
    def __init__(
            self, kernel: KernelFunction, mean: MeanFunction = None,
            initial_noise_variance=None, encoding_type=None,
            has_covariance_scale=False, **kwargs):
        super(MarginalLikelihood, self).__init__(**kwargs)
        if mean is None:
            mean = ScalarMeanFunction()
        if initial_noise_variance is None:
            initial_noise_variance = INITIAL_NOISE_VARIANCE
        if encoding_type is None:
            encoding_type = DEFAULT_ENCODING
        self.encoding_noise = create_encoding(
             encoding_type, initial_noise_variance, NOISE_VARIANCE_LOWER_BOUND,
             NOISE_VARIANCE_UPPER_BOUND, 1, Gamma(mean=0.1, alpha=0.1))
        self.mean = mean
        self.kernel = kernel
        if has_covariance_scale:
            self.encoding_covscale = create_encoding(
                encoding_type, INITIAL_COVARIANCE_SCALE,
                COVARIANCE_SCALE_LOWER_BOUND, COVARIANCE_SCALE_UPPER_BOUND, 1,
                LogNormal(0.0, 1.0))
        with self.name_scope():
            self.noise_variance_internal = register_parameter(
                self.params, 'noise_variance', self.encoding_noise)
            if has_covariance_scale:
                self.covariance_scale_internal = register_parameter(
                    self.params, 'covariance_scale', self.encoding_covscale)

    def _noise_variance(self):
        return encode_unwrap_parameter(
            self.noise_variance_internal, self.encoding_noise)

    def _covariance_scale(self):
        if self.encoding_covscale is not None:
            return encode_unwrap_parameter(
                self.covariance_scale_internal, self.encoding_covscale)
        else:
            return 1.0

    def get_posterior_state(self, features, targets):
        if self.encoding_covscale is not None:
            kernel = (self.kernel, self._covariance_scale())
        else:
            kernel = self.kernel
        return GaussProcPosteriorState(
            features=features,
            targets=targets,
            mean=self.mean,
            kernel=kernel,
            noise_variance=self._noise_variance())

    def forward(self, features, targets):
        """
        Actual computation of the marginal likelihood
        See http://www.gaussianprocess.org/gpml/chapters/RW.pdf, equation (2.30)

        :param features: input data matrix X of size (n, d)
        :param targets: targets corresponding to X, of size (n, 1)
        """
        return self.get_posterior_state(features, targets).neg_log_likelihood()

    def param_encoding_pairs(self):
        """
        Return a list of tuples with the Gluon parameters of the likelihood and their respective encodings
        """
        own_param_encoding_pairs = [
            (self.noise_variance_internal, self.encoding_noise)]
        if self.encoding_covscale is not None:
            own_param_encoding_pairs.append(
                (self.covariance_scale_internal, self.encoding_covscale))
        return own_param_encoding_pairs + self.mean.param_encoding_pairs() + \
               self.kernel.param_encoding_pairs()

    def box_constraints_internal(self):
        """
        Collect the box constraints for all the underlying parameters
        """
        all_box_constraints = {}
        for param, encoding in self.param_encoding_pairs():
            assert encoding is not None,\
                "encoding of param {} should not be None".format(param.name)
            all_box_constraints.update(encoding.box_constraints_internal(param))
        return all_box_constraints

    def get_noise_variance(self, as_ndarray=False):
        noise_variance = self._noise_variance()
        return noise_variance if as_ndarray else anp.reshape(noise_variance, (1,))[0]

    def set_noise_variance(self, val):
        self.encoding_noise.set(self.noise_variance_internal, val)
        
    def get_covariance_scale(self):
        if self.encoding_covscale is not None:
            return self._covariance_scale()[0]
        else:
            return 1.0

    def set_covariance_scale(self, covariance_scale):
        assert self.encoding_covscale is not None, \
            "covariance_scale is fixed to 1"
        self.encoding_covscale.set(
            self.covariance_scale_internal, covariance_scale)

    def get_params(self):
        result = {'noise_variance': self.get_noise_variance()}
        if self.encoding_covscale is not None:
            result['covariance_scale'] = self.get_covariance_scale()
        for pref, func in [('kernel_', self.kernel), ('mean_', self.mean)]:
            result.update({
                (pref + k): v for k, v in func.get_params().items()})
        return result
    
    def set_params(self, param_dict):
        for pref, func in [('kernel_', self.kernel), ('mean_', self.mean)]:
            len_pref = len(pref)
            stripped_dict = {
                k[len_pref:]: v for k, v in param_dict.items()
                if k.startswith(pref)}
            func.set_params(stripped_dict)
        self.set_noise_variance(param_dict['noise_variance'])
        if self.encoding_covscale is not None:
            self.set_covariance_scale(param_dict['covariance_scale'])
