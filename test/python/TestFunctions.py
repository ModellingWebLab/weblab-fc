
import unittest

import fc.language.expressions as E
import fc.language.statements as S
import fc.language.values as V
import fc.utility.environment as Env
from fc.utility.error_handling import ProtocolError

N = E.N


class TestFunctions(unittest.TestCase):
    """Test lambda expressions and all function methods."""

    def TestFuncDefinitions(self):
        env = Env.Environment()
        parameters = ["a", "b"]
        body = [S.Return(E.NameLookUp("b"), E.NameLookUp("a"))]
        swap = E.LambdaExpression(parameters, body)
        env.ExecuteStatements([S.Assign(["swap"], swap)])
        args = [N(1), N(2)]
        swap_call = E.FunctionCall("swap", args)
        result = swap_call.Evaluate(env)
        self.assert_(isinstance(result, V.Tuple))
        self.assertAlmostEqual(result.values[0].value, 2)
        self.assertAlmostEqual(result.values[1].value, 1)
        env.ExecuteStatements([S.Assign(parameters, swap_call)])
        defined = env.DefinedNames()
        self.assertEqual(len(defined), 3)
        self.assertEqual(env.LookUp('a').value, 2)
        self.assertEqual(env.LookUp('b').value, 1)

        args = [N(3), N(5)]
        swap_call = E.FunctionCall("swap", args)
        result = swap_call.Evaluate(env)
        self.assert_(isinstance(result, V.Tuple))
        self.assertAlmostEqual(result.values[0].value, 5)
        self.assertAlmostEqual(result.values[1].value, 3)

    def TestLambdaExpressionWrap(self):
        env = Env.Environment()
        add = E.LambdaExpression.Wrap(E.Plus, 3)
        args = [N(1), N(2), N(3)]
        add_call = E.FunctionCall(add, args)
        result = add_call.Evaluate(env)
        self.assertEqual(result.value, 6)

    def TestNestedFunction(self):
        env = Env.Environment()
        nested_body = [S.Return(E.Plus(E.NameLookUp('input'), E.NameLookUp('outer_var')))]
        nested_function = E.LambdaExpression(["input"], nested_body)
        body = [S.Assign(["nested_fn"], nested_function),
                S.Assign(["outer_var"], N(1)),
                S.Return(E.Eq(E.FunctionCall("nested_fn", [N(1)]), N(2)))]
        nested_scope = E.LambdaExpression([], body)
        nested_call = E.FunctionCall(nested_scope, [])
        result = nested_call.Evaluate(env)
        self.assertEqual(result.value, 1)

    def TestFunctionsWithDefaultsUsed(self):
        # Function has default which is used
        env = Env.Environment()
        nested_body = [S.Return(E.Plus(E.NameLookUp('input'), E.NameLookUp('outer_var')))]
        nested_function = E.LambdaExpression(["input"], nested_body, defaultParameters=[V.Simple(1)])
        body = [S.Assign(["nested_fn"], nested_function),
                S.Assign(["outer_var"], N(1)),
                S.Return(E.Eq(E.FunctionCall("nested_fn", [E.Const(V.DefaultParameter())]), N(2)))]
        nested_scope = E.LambdaExpression([], body)
        nested_call = E.FunctionCall(nested_scope, [])
        result = nested_call.Evaluate(env)
        self.assertEqual(result.value, 1)

    def TestFunctionsWithDefaultsUnused(self):
        # Function has default, but value is explicitly assigned in this case
        env = Env.Environment()
        nested_body = [S.Return(E.Plus(E.NameLookUp('input'), E.NameLookUp('outer_var')))]
        nested_function = E.LambdaExpression(["input"], nested_body, defaultParameters=[V.Simple(0)])
        body = [S.Assign(["nested_fn"], nested_function),
                S.Assign(["outer_var"], N(1)),
                S.Return(E.Eq(E.FunctionCall("nested_fn", [N(1)]), N(2)))]
        nested_scope = E.LambdaExpression([], body)
        nested_call = E.FunctionCall(nested_scope, [])
        result = nested_call.Evaluate(env)
        self.assertEqual(result.value, 1)

    def TestMultipleDefaultValues(self):
        env = Env.Environment()
        parameters = ['a', 'b', 'c']
        body = [S.Return(E.Plus(E.NameLookUp('a'), E.NameLookUp('b'), E.NameLookUp('c')))]
        add = E.LambdaExpression(parameters, body, defaultParameters=[V.Simple(1), V.Simple(2), V.Simple(3)])
        args = [E.Const(V.DefaultParameter()), E.Const(V.DefaultParameter()), E.Const(V.DefaultParameter())]
        add_call = E.FunctionCall(add, args)
        result = add_call.Evaluate(env)
        self.assertEqual(result.value, 6)

        args = [N(3)]
        add_call = E.FunctionCall(add, args)
        result = add_call.Evaluate(env)
        self.assertEqual(result.value, 8)

        args = [E.Const(V.DefaultParameter()), E.Const(V.DefaultParameter()), N(1)]
        add_call = E.FunctionCall(add, args)
        result = add_call.Evaluate(env)
        self.assertEqual(result.value, 4)

        args = [N(4), E.Const(V.DefaultParameter()), N(4)]
        add_call = E.FunctionCall(add, args)
        result = add_call.Evaluate(env)
        self.assertEqual(result.value, 10)

    def TestAssertStatement(self):
        env = Env.Environment()
        # evaluates to one, assertion should pass
        env.ExecuteStatements([S.Assert(N(1))])
        # evaluates to zero, assertion should fail
        self.assertRaises(ProtocolError, env.ExecuteStatements, [S.Assert(N(0))])
        # evaluates to non-simple value, assertion should fail
        self.assertRaises(ProtocolError, env.ExecuteStatements, [S.Assert(E.Const(V.DefaultParameter()))])
