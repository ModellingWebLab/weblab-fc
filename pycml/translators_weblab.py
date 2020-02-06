
class CellMLToCythonTranslator(CellMLToPythonTranslator):
    """Output Cython code suitable for the Python implementation of Functional Curation.

    Unlike the base class, code generated by this translator can't inherit from a pure Python base class.
    It also hardcodes using our Cython wrapper of CVODE as the solver.

    Note that we use 2 types of vector in the generated code: numpy arrays with the same names as for
    CellMLToPythonTranslator provide the same interface to the FC python code, and N_Vector views on the
    same memory provide fast access for the ODE solver.
    """

    USES_SUBSIDIARY_FILE = True
#     TYPE_VECTOR = 'cdef Sundials.N_Vector'
#     TYPE_VECTOR_REF = 'cdef Sundials.N_Vector'
    TYPE_DOUBLE = 'cdef double '
    TYPE_CONST_DOUBLE = 'cdef double '

    def output_file_name(self, model_filename):
        """Generate a name for our output file, based on the input file."""
        return os.path.splitext(model_filename)[0] + '.pyx'

    def subsidiary_file_name(self, output_filename):
        """Our subsidiary file is the setup.py used to build the extension."""
        return output_filename, os.path.join(os.path.dirname(output_filename), 'setup.py')

#     def vector_index(self, vector, i):
#         """Return code for accessing the i'th index of vector."""
#         return '(<Sundials.N_VectorContent_Serial>(' + vector + ').content).data[' + str(i) + ']'
#
#     def vector_create(self, vector, size):
#         """Return code for creating a new vector with the given size."""
#         return ''.join(map(str, [self.TYPE_VECTOR, vector, self.EQ_ASSIGN,
#                                  'Sundials.N_VNew_Serial(', size, ')', self.STMT_END]))
#
#     def vector_initialise(self, vector, size):
#         """Return code for creating an already-declared vector with the given size."""
#         return ''.join(map(str, [vector, self.EQ_ASSIGN, 'Sundials.N_VNew_Serial(', size, ')', self.STMT_END]))

    def output_assignment(self, expr):
        """Output an assignment statement.

        Avoids most of the magic in the Chaste version of this method, except for handling parameters specially.
        """
        if isinstance(expr, cellml_variable) and expr in self.cell_parameters:
            return
        return CellMLTranslator.output_assignment(self, expr)

    def output_top_boilerplate(self):
        """Output file content occurring before the model equations: basically just imports in this case.

        The main RHS 'method' is actually a plain function so we can use it as a C callback.
        """
        self.analyse_model()
        self.write_setup_py()
        # Start file output
        self.writeln('# cython: profile=True')
        self.output_common_imports()
        self.writeln('cimport libc.math as math')
        self.writeln('cimport numpy as np')
        self.writeln('import os')
        self.writeln('import shutil')
        self.writeln('import sys')
        self.writeln()
        self.writeln('from fc.sundials.solver cimport CvodeSolver')
        self.writeln('cimport fc.sundials.sundials as Sundials')
        self.writeln('from fc.utility.error_handling import ProtocolError')
        self.writeln()
        self.output_data_tables()

    def output_bottom_boilerplate(self):
        """Output file content occurring after the model equations, i.e. the model class."""
        base_class = 'CvodeSolver'
        self.writeln('cdef class ', self.class_name, '(', base_class, '):')
        self.open_block()
        # Declare member attributes. Note that state and _state come from the base class.
        self.writeln('cdef public char* freeVariableName')
        self.writeln('cdef public double freeVariable')
        self.writeln('cdef public object stateVarMap')
        self.writeln('cdef public np.ndarray initialState')
        self.writeln('cdef public object parameterMap')
        self.writeln('cdef public np.ndarray parameters')
        self.writeln('cdef public object outputNames')
        self.writeln()
        self.writeln('cdef public object savedStates')
        self.writeln('cdef public object env')
        self.writeln('cdef public bint dirty')
        self.writeln('cdef public char* outputPath')
        self.writeln('cdef public object indentLevel')
        self.writeln()
        self.writeln('cdef public object _module')
        self.writeln('cdef public object simEnv')
        self.writeln()
        self.writeln('cdef Sundials.N_Vector _parameters')
        self.writeln('cdef public object _outputs')
        self.writeln()
        # Constructor
        self.writeln('def __init__(self):')
        self.open_block()
        self.output_common_constructor_content()
        self.writeln('self.state = self.initialState.copy()')
        self.writeln('self.savedStates = {}')
        self.writeln('self.dirty = False')
        self.writeln('self.indentLevel = 0')
        self.writeln('self.AssociateWithModel(self)')
        self.writeln('self._parameters = Sundials.N_VMake_Serial(len(self.parameters), <Sundials.realtype*>(<np.ndarray>self.parameters).data)')
        # TODO: Use a separate environment for each ontology
        self.writeln('self.env = Env.ModelWrapperEnvironment(self)')
        # Initialise CVODE
        self.close_block()
        self.writeln('def SetRhsWrapper(self):')
        self.open_block()
        self.writeln('flag = Sundials.CVodeInit(self.cvode_mem, _EvaluateRhs, 0.0, self._state)')
        self.writeln('self.CheckFlag(flag, "CVodeInit")')
        self.close_block()
        # Cython-level destructor
        self.writeln('def __dealloc__(self):')
        self.open_block()
        self.writeln('if self._parameters != NULL:')
        self.writeln('    Sundials.N_VDestroy_Serial(self._parameters)')
        self.close_block()
        # Methods to match the AbstractModel class
        self.writeln('def SetOutputFolder(self, path):')
        self.open_block()
        self.writeln("if os.path.isdir(path) and path.startswith('/tmp'):")
        self.writeln('shutil.rmtree(path)', indent_offset=1)
        self.writeln('os.mkdir(path)')
        self.writeln('self.outputPath = path')
        self.close_block()
        self.writeln('def SetIndentLevel(self, indentLevel):')
        self.open_block()
        self.writeln('self.indentLevel = indentLevel')
        self.close_block()
        # Methods to match the AbstractOdeModel class
        self.writeln('def SetSolver(self, solver):')
        self.open_block()
        self.writeln('print >>sys.stderr, "  " * self.indentLevel, "SetSolver: Models implemented using Cython contain a built-in ODE solver, so ignoring setting."')
        self.close_block()
        self.writeln('def GetEnvironmentMap(self):')
        self.open_block()
        self.writeln('return {', nl=False)
        # TODO: Use a separate env for each ontology
        for i, prefix in enumerate(self.model._cml_protocol_namespaces.iterkeys()):
            if i > 0:
                self.write(', ')
            self.write("'%s': self.env" % prefix)
        self.writeln('}', indent=False)
        self.close_block()
        self.writeln('cpdef SetFreeVariable(self, double t):')
        self.open_block()
        self.writeln('self.freeVariable = t')
        self.writeln(base_class, '.SetFreeVariable(self, t)')
        self.close_block()
        self.writeln('def SaveState(self, name):')
        self.open_block()
        self.writeln('self.savedStates[name] = self.state.copy()')
        self.close_block()
        self.writeln('cpdef ResetState(self, name=None):')
        self.open_block()
        self.writeln('if name is None:')
        self.writeln(base_class, '.ResetSolver(self, self.initialState)', indent_offset=1)
        self.writeln('else:')
        self.writeln(base_class, '.ResetSolver(self, self.savedStates[name])', indent_offset=1)
        self.close_block()
        self.writeln('cpdef GetOutputs(self):')
        self.open_block()
        self.writeln('cdef np.ndarray[Sundials.realtype, ndim=1] parameters = self.parameters')
        self.param_vector_name = 'parameters'
        self.TYPE_CONST_UNSIGNED = 'cdef unsigned '
        self.output_get_outputs_content()
        del self.TYPE_CONST_UNSIGNED
        self.param_vector_name = 'self.parameters'
        self.close_block()

    def output_mathematics(self):
        """Output the mathematics in this model.

        This generates the ODE right-hand side function, "EvaluateRhs(self, t, y)", but as a C-style callback for CVODE.
        """
        self.writeln('cdef int _EvaluateRhs(Sundials.realtype ', self.code_name(self.free_vars[0]),
                     ', Sundials.N_Vector y, Sundials.N_Vector ydot, void* user_data):')
        self.open_block()
        self.writeln('model = <object>user_data')
        self.writeln('cdef np.ndarray[Sundials.realtype, ndim=1] parameters = <np.ndarray>model.parameters')
        self.param_vector_name = 'parameters'
        # Work out what equations are needed to compute the derivatives
        derivs = set(map(lambda v: (v, self.free_vars[0]), self.state_vars))
        nodeset = self.calculate_extended_dependencies(derivs)
        # Code to do the computation
        self.output_comment('State variables')
        for i, var in enumerate(self.state_vars):
            if var in nodeset:
                self.writeln(self.TYPE_DOUBLE, self.code_name(var), ' = (<Sundials.N_VectorContent_Serial>y.content).data[', i, ']')
        self.writeln()
        self.TYPE_CONST_UNSIGNED = 'cdef unsigned '
        nodes_used = self.output_data_table_lookups(nodeset)
        del self.TYPE_CONST_UNSIGNED
        self.writeln()
        self.output_comment('Mathematics')
        self.output_equations(nodeset - nodes_used)
        self.writeln()
        # Assign to derivatives vector
        for i, var in enumerate(self.state_vars):
            self.writeln('(<Sundials.N_VectorContent_Serial>ydot.content).data[', i, '] = ', self.code_name(var, True))
        self.param_vector_name = 'self.parameters'
        self.close_block()

    def output_array_definition(self, array_name, array_data):
        """Output code to create and fill a fixed-size 1d array."""
        self.writeln('cdef Sundials.realtype[', len(array_data), ']', array_name)
        self.writeln(array_name, '[:] = [', ', '.join(map(lambda f: "%.17g" % f, array_data)), ']')

    def fast_floor(self, arg):
        """Return code to compute the floor of an argument as an integer quickly, typically by casting."""
        return "int(%s)" % arg

    def write_setup_py(self):
        """Write our subsidiary setup.py file for building the extension."""
        self.out2.write("""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules=[
    Extension("%(filebase)s",
              ["%(filebase)s.pyx"],
              include_dirs=[numpy.get_include(), '%(fcpath)s'],
              #library_dirs=['%(fcpath)s/fc/sundials'],
              libraries=['sundials_cvode', 'sundials_nvecserial', 'm'])
              # users can set CFLAGS and LDFLAGS in their env if needed
]

setup(
  name = "%(filebase)s",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules
)
""" % {'filebase': os.path.splitext(os.path.basename(self.output_filename))[0],
       'fcpath': os.path.join(os.path.dirname(__file__), '../../projects/FunctionalCuration/src/python')})
