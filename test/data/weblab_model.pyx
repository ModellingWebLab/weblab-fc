# cython: profile=True
#
#
# Web Lab model hodgkin_huxley_squid_axon_model_1952_modified
#
# Generated by fccodegen 0.0.1 on 2018-12-06 11:12:42
#
#
#
cimport fc.sundials.sundials as Sundials

cimport libc.math as math
cimport numpy as np

import numpy as np
import os
import shutil
import sys

import fc.simulations.model as Model
import fc.utility.environment as Env
import fc.language.values as V
from fc.sundials.solver cimport CvodeSolver
from fc.utility.error_handling import ProtocolError


cdef int _EvaluateRhs(Sundials.realtype var_time,
                      Sundials.N_Vector y,
                      Sundials.N_Vector ydot,
                      void* user_data):
    """
    Cython wrapper around a model RHS that uses numpy, for calling by CVODE.

    See :meth:`fc.sundials.solver._EvaluateRhs()`.
    """
    # We passed the Python model object in as CVODE user data; get it back as an object
    model = <object>user_data
    cdef np.ndarray[Sundials.realtype, ndim=1] parameters = <np.ndarray>model.parameters

    # Unpack state variables
    cdef double var_V = (<Sundials.N_VectorContent_Serial>y.content).data[0]
    cdef double var_m = (<Sundials.N_VectorContent_Serial>y.content).data[1]
    cdef double var_h = (<Sundials.N_VectorContent_Serial>y.content).data[2]
    cdef double var_n = (<Sundials.N_VectorContent_Serial>y.content).data[3]

    # Mathematics
    cdef double var_g_L = 0.3
    cdef double var_Cm = 1.0
    cdef double var_E_R = -75.0
    cdef double var_E_L = 10.613 + var_E_R
    cdef double var_i_L = (-var_E_L + var_V) * var_g_L
    cdef double var_stim_amplitude = -20.0
    cdef double var_stim_duration = 0.5
    cdef double var_stim_end = 10000.0
    cdef double var_stim_period = 1000.0
    cdef double var_stim_start = 10.0
    cdef double var_i_Stim = ((var_stim_amplitude) if (var_time >= var_stim_start and var_time <= var_stim_end and -var_stim_start - var_stim_period * math.floor((-var_stim_start + var_time) / var_stim_period) + var_time <= var_stim_duration) else (0.0))
    cdef double var_E_K = -12.0 + var_E_R
    cdef double var_g_K = parameters[1]
    cdef double var_alpha_n = -0.01 * (65.0 + var_V) / (-1.0 + 0.0015034391929775724 * math.exp(-0.1 * var_V))
    cdef double var_beta_n = 0.31919868225786585 * math.exp(0.0125 * var_V)
    cdef double d_dt_n = (1.0 - var_n) * var_alpha_n - var_beta_n * var_n
    cdef double var_i_K = var_n**4.0 * (-var_E_K + var_V) * var_g_K
    cdef double var_E_Na = 115.0 + var_E_R
    cdef double var_g_Na = parameters[0]
    cdef double var_alpha_h = 0.0016462422099206377 * math.exp(-0.05 * var_V)
    cdef double var_beta_h = 1.0 / (1.0 + 0.011108996538242306 * math.exp(-0.1 * var_V))
    cdef double d_dt_h = (1.0 - var_h) * var_alpha_h - var_beta_h * var_h
    cdef double var_alpha_m = -0.1 * (50.0 + var_V) / (-1.0 + 0.006737946999085467 * math.exp(-0.1 * var_V))
    cdef double var_beta_m = 0.06201541439603731 * math.exp(-0.05555555555555555 * var_V)
    cdef double d_dt_m = (1.0 - var_m) * var_alpha_m - var_beta_m * var_m
    cdef double var_i_Na = var_m**3.0 * (-var_E_Na + var_V) * var_g_Na * var_h
    cdef double d_dt_V = (-var_i_L - var_i_Stim - var_i_K - var_i_Na) / var_Cm

    # Pack state variable derivatives
    (<Sundials.N_VectorContent_Serial>ydot.content).data[0] = d_dt_V
    (<Sundials.N_VectorContent_Serial>ydot.content).data[1] = d_dt_m
    (<Sundials.N_VectorContent_Serial>ydot.content).data[2] = d_dt_h
    (<Sundials.N_VectorContent_Serial>ydot.content).data[3] = d_dt_n


cdef class TestModel(CvodeSolver):

    # The name of the free variable, for use in the ModelWrapperEnvironment
    # From: fc.simulations.AbstractOdeModel
    cdef public char* freeVariableName

    # The value of the free variable
    # From: fc.simulations.AbstractOdeModel
    cdef public double freeVariable

    # A mapping from variable name to index within the state variable vector,
    # for use in the ModelWrapperEnvironment
    # From: fc.simulations.AbstractOdeModel
    cdef public object stateVarMap

    # A numpy array containing initial values for the state variables
    # From: fc.simulations.AbstractOdeModel
    cdef public np.ndarray initialState

    # A mapping from parameter name to index within the parameters vector, for
    # use in the ModelWrapperEnvironment
    # From: fc.simulations.AbstractOdeModel
    cdef public object parameterMap

    # A numpy array containing model parameter values
    # From: fc.simulations.AbstractOdeModel
    cdef public np.ndarray parameters

    # An ordered list of the names of the model outputs, as they will be
    # returned by GetOutputs
    # From: fc.simulations.AbstractOdeModel
    cdef public object outputNames

    # Mapping from names to saved model states.
    # From: fc.simulations.AbstractOdeModel
    cdef public object savedStates

    # Maps oxmeta variable names to model variables (outputs, states,
    # parameters, or the free variable).
    # From: fc.simulations.AbstractOdeModel
    # See: fc.utility.environment.ModelWrapperEnvironment
    cdef public object env

    # True if the solver needs to be reset due to a model change made in the
    # ModelWrapperEnvironment.
    # From: fc.simulations.AbstractOdeModel
    cdef public bint dirty

    # Environment for the simulation running this model. Mainly useful when
    # evaluating set_variable() type modifiers during the course of a
    # simulation.

    # Where to write protocol outputs, error logs, etc.
    # From: fc.simulations.AbstractModel
    cdef public char* outputPath

    # Level of indentation to use for progress output.
    # From: fc.simulations.AbstractModel
    cdef public object indentLevel

    # Link to generated module.
    # Set in: fc.utility.protocol.Protocol
    # Note: Nobody seems to ever access this variable. Seems this is just to
    # prevent garbage collection.
    cdef public object _module

    # NOT SURE
    cdef public object simEnv

    # Cached list of output values (single values or vectors e.g. the state) to
    # avoid recreating a list every time output is returned.
    cdef public object _outputs

    # Seems to be unused at the moment
    #cdef Sundials.N_Vector _parameters

    def __init__(self):
        self.freeVariableName = "time"
        self.freeVariable = 0.0

        # State values
        self.state = np.zeros(4)

        # Mapping from oxmeta names to state indices; only for states that have
        # a variable name.
        self.stateVarMap = {}

        # Initial state
        self.initialState = np.zeros(4)
        self.initialState[0] = -75.0
        self.initialState[1] = 0.05
        self.initialState[2] = 0.6
        self.initialState[3] = 0.325

        # Mapping of parameter oxmeta names to parameter array indices
        self.parameterMap = {}
        self.parameterMap['membrane_fast_sodium_current_conductance'] = 0
        self.parameterMap['membrane_potassium_current_conductance'] = 1

        # Initial parameter values
        self.parameters = np.zeros(2)
        self.parameters[0] = 120.0
        self.parameters[1] = 36.0

        # Oxmeta names of output variables
        self.outputNames = []
        self.outputNames.append('membrane_fast_sodium_current')
        self.outputNames.append('membrane_voltage')
        self.outputNames.append('time')

        # Create and cache list of arrays, to avoid constant list/array
        # creation
        self._outputs = []
        self._outputs.append(np.array(0.0))
        self._outputs.append(np.array(0.0))
        self._outputs.append(np.array(0.0))
        # TODO Handle vector outputs

        self.state = self.initialState.copy()
        self.savedStates = {}
        self.dirty = False
        self.indentLevel = 0
        self.AssociateWithModel(self)
        #self._parameters = Sundials.N_VMake_Serial(
        #    len(self.parameters),
        #    <Sundials.realtype*>(<np.ndarray>self.parameters).data
        #)
        self.env = Env.ModelWrapperEnvironment(self)

    #def __dealloc__(self):
    #    if self._parameters != NULL:
    #        Sundials.N_VDestroy_Serial(self._parameters)

    def GetEnvironmentMap(self):
        """
        Get a map from ontology prefix to the environment containing model
        variables annotated with that ontology.

        See :meth:`fc.simulations.AbstractOdeModel.GetEnvironmentMap()`.
        """
        # TODO Some part of this might need to be generated
        return {
            'pycml': self.env,
            'cmeta': self.env,
            'cg': self.env,
            'csub': self.env,
            'cs': self.env,
            'oxmeta': self.env,
            'lut': self.env,
            'proto': self.env,
            'None': self.env,
            'bqs': self.env,
            'pe': self.env,
            'dcterms': self.env,
            'xml': self.env,
            'dc': self.env,
            'bqbiol': self.env,
            'cml': self.env,
            'solver': self.env,
            'doc': self.env,
            'm': self.env,
            'rdf': self.env,
            'cellml': self.env,
            'vCard': self.env,
        }

    cpdef GetOutputs(self):
        """
        Return a list of the model's outputs at its current state.

        NB: this should return a Python list containing the model outputs as
        numpy arrays, not subclasses of V.AbstractValue.
        The order of outputs in this list must match self.outputNames, a list
        of the output names, which must be set by subclass constructors.

        See :meth:`fc.simulations.AbstractModel.getOutputs()`.
        """

        # Get parameters as sundials realtype numpy array
        cdef np.ndarray[Sundials.realtype, ndim=1] parameters = self.parameters

        # Get current free variable
        cdef double var_time = self.freeVariable

        # Unpack state variables
        cdef double var_V = self.state[0]
        cdef double var_m = self.state[1]
        cdef double var_h = self.state[2]
        cdef double var_n = self.state[3]

        # Mathematics
        cdef double var_E_R = -75.0
        cdef double var_E_Na = 115.0 + var_E_R
        cdef double var_g_Na = parameters[0]
        cdef double var_i_Na = var_m**3.0 * (-var_E_Na + var_V) * var_g_Na * var_h

        # Update output vector and return
        outputs = self._outputs
        outputs[0][()] = var_i_Na
        outputs[1][()] = var_V
        outputs[2][()] = var_time
        return outputs

    cpdef ResetState(self, name=None):
        """
        Reset the model to the given named saved state, or to initial
        conditions if no name given.

        See :meth:`fc.simulations.AbstractOdeModel.ResetState()`.
        """
        if name is None:
            CvodeSolver.ResetSolver(self, self.initialState)
        else:
            CvodeSolver.ResetSolver(self, self.savedStates[name])

    def SaveState(self, name):
        """
        Save a copy of the current model state associated with the given name,
        to be restored using :meth:`ResetState()`.

        See :meth:`fc.simulations.AbstractOdeModel.SaveState()`.
        """
        self.savedStates[name] = self.state.copy()

    cpdef SetFreeVariable(self, double t):
        """
        Set the value of the free variable (typically time), but retain the
        model's current state.

        See :meth:`fc.simulations.AbstractOdeModel.SetFreeVariable()`.
        """
        self.freeVariable = t
        CvodeSolver.SetFreeVariable(self, t)

    def SetIndentLevel(self, indentLevel):
        """
        Set the level of indentation to use for progress output.

        See :meth:`fc.simulations.AbstractModel.setIndentLevel()`.
        """
        self.indentLevel = indentLevel

    def SetOutputFolder(self, path):
        # TODO This is undocumented in fc
        if os.path.isdir(path) and path.startswith('/tmp'):
            shutil.rmtree(path)
        os.mkdir(path)
        self.outputPath = path

    def SetRhsWrapper(self):
        flag = Sundials.CVodeInit(
            self.cvode_mem, _EvaluateRhs, 0.0, self._state)
        self.CheckFlag(flag, 'CVodeInit')

    def SetSolver(self, solver):
        """
        Specify the ODE solver to use for this model.

        See :meth:`fc.simulations.AbstractOdeModel.SetSolver()`.
        """
        # TODO Update this (and rest of fc) to Python3
        # TODO Use logging here, or raise an exception
        print >>sys.stderr, '  ' * self.indentLevel, 'SetSolver: Models implemented using Cython contain a built-in ODE solver, so ignoring setting.'

