#include "postgres.h"
#include "fmgr.h"

#include "utils/builtins.h"
#include "executor/executor.h"
#include "utils/lsyscache.h"

#include <Python.h>
#include <pg_config.h>

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(test_function);
PG_FUNCTION_INFO_V1(call_python_udf_from_c);

typedef enum
{
    RES_UNDEFINED,
    RES_ERR_MODULE,
    RES_ERR_CLASS,
    RES_ERR_FUNC,
    RES_ERR_CALL,
    RES_SUCCESS
} RESULT_CODE;


/* Prototype of our hook */
void aida_ExecutorEnd_hook(QueryDesc *queryDesc);
PyObject *getFuncObject(const char* funcName, RESULT_CODE *result);
void *aida_loop(void *arg);
void print_cached_modules();
void resToString( RESULT_CODE eRes, char buffer[50] );

void *aida_loop(void *arg)
{
    PyObject *pResult = NULL, *pFunc = arg, *pArgs = NULL;

    Py_Initialize();

    pArgs = PyTuple_Pack(1, PyLong_FromLong(10)); // temp dummy argument
    pResult = PyObject_CallObject(pFunc, NULL);
    if (pResult) {
        ereport(INFO, (errmsg("Success calling UDF()")));
        //Py_DECREF(pResult);
    }
    else {
        ereport(INFO, (errmsg("Failure calling UDF()")));
    }

    return NULL;
}

/* Pointer to old hook */
ExecutorEnd_hook_type prev_ExecutorEnd_hook;

void _PG_init(void)
{

    ereport(INFO, (errmsg("hook startup")));

    /* Store the old hook */
    prev_ExecutorEnd_hook = ExecutorEnd_hook;
    /* Change the hook to the new function */
    ExecutorEnd_hook = aida_ExecutorEnd_hook;
}



bool threadStarted = false; 


PyObject *getFuncObject(const char* funcName, RESULT_CODE *result)
{
    PyObject *pModule = NULL, *pClass = NULL, *pFunc = NULL;

    if ( NULL == result ) return NULL;

    PyRun_SimpleString("import aida.aida");  // preloading module

    //pModule = PyImport_ImportModule("aida.aida");
    pModule = PyImport_GetModule(PyUnicode_DecodeFSDefault("aida.aida"));
    if (pModule != NULL) {

        ereport(INFO, (errmsg("Module type: %s", Py_TYPE(pModule)->tp_name)));

        //Py_XDECREF(pDir);  // Free the list

    } else {
        //PyImport_ImportModule("aida.aida");
        *result = RES_ERR_MODULE;
        goto cleanup;
    }

    //Py_DECREF(pModule);

    // get the class
    pClass = PyObject_GetAttrString(pModule, "AIDA");
    if (!pClass) {
        *result = RES_ERR_CLASS;
        goto cleanup;
    }

    //Py_DECREF(pClass);

    // get the static method
    pFunc = PyObject_GetAttrString(pClass, "udf");
    if (!pFunc || !PyCallable_Check(pFunc)) {
        *result = RES_ERR_FUNC;
        goto cleanup;
    }

    *result = RES_SUCCESS;

cleanup:
    return pFunc;
}

void resToString( RESULT_CODE eRes, char buffer[50] )
{
    char *msg = NULL;
    switch ( eRes )
    {
        case RES_ERR_MODULE:
            msg = "Could not import module.";
            break;
        case RES_ERR_CLASS:
            msg = "Could not find class.";
            break;
        case RES_ERR_FUNC:
            msg = "Could not find method.";
            break;
        case RES_ERR_CALL:
            msg = "Method call failed.";
            break;
        case RES_SUCCESS:
            msg = "Success.";
            break;
        default:
            msg = "Undefined.";
            break;
    }

    memset(buffer, 0, sizeof( buffer) );
    strncpy(buffer, msg, sizeof( buffer ) - 1);
}

// Function to call Python UDF from C
Datum call_python_udf_from_c(PG_FUNCTION_ARGS) {
    RESULT_CODE resultCode = RES_UNDEFINED;
    char msg[50];
    PyObject *pFunc = NULL, *pArgs = NULL, *pResult = NULL;
    
    Py_Initialize();

    // if ( !threadStarted )
    // {
    //     RESULT_CODE resultCode = RES_UNDEFINED;
    //     PyObject *pUDFFunc = NULL;

    //     pUDFFunc = getFuncObject( "udf", &resultCode );

    //     if ( resultCode == RES_SUCCESS )
    //     {
    //         pthread_t thread;
    //         pthread_create(&thread, NULL, aida_loop, pUDFFunc);
    //         pthread_detach(thread);
    //         threadStarted = true;
    //     }
    // }

    pFunc = getFuncObject( "udf", &resultCode );

    if ( RES_SUCCESS == resultCode )
    {
        //4. Call the static method with an argument (e.g., 10)
        pArgs = PyTuple_Pack(1, PyLong_FromLong(10));
        pResult = PyObject_CallObject(pFunc, pArgs);
        if (pResult) {
            long result = PyLong_AsLong(pResult);
            ereport(INFO, (errmsg("Result from callback(): %ld", result)));
            //Py_DECREF(pResult);
        } else {
            resultCode = RES_ERR_CALL;
        }
    }
    
    resToString( resultCode, msg );

    // Finalize Python Interpreter
    //Py_Finalize();

    PG_RETURN_TEXT_P(cstring_to_text(msg));
}

Datum test_function(PG_FUNCTION_ARGS)
{
    const char *test = "Test from AIDA";
    PG_RETURN_TEXT_P(cstring_to_text(test));
}

void aida_ExecutorEnd_hook(QueryDesc *queryDesc)
{
    /* Call the previous hook */
    if (prev_ExecutorEnd_hook)
    {
        prev_ExecutorEnd_hook(queryDesc);
    }
}

// void print_cached_modules() {
//     PyObject *sys_module, *modules_dict, *keys_list;

//     // Import the sys module
//     sys_module = PyImport_ImportModule("sys");
//     if (!sys_module) {
//         PyErr_Print();
//         fprintf(stderr, "Failed to import sys module\n");
//         return;
//     }

//     // Get sys.modules dictionary
//     modules_dict = PyObject_GetAttrString(sys_module, "modules");
//     Py_DECREF(sys_module);  // sys_module no longer needed

//     if (!modules_dict || !PyDict_Check(modules_dict)) {
//         PyErr_Print();
//         fprintf(stderr, "Failed to get sys.modules\n");
//         Py_XDECREF(modules_dict);
//         return;
//     }

//     // Get keys (module names)
//     keys_list = PyDict_Keys(modules_dict);
//     Py_DECREF(modules_dict);  // modules_dict no longer needed

//     if (keys_list && PyList_Check(keys_list)) {
//         Py_ssize_t size = PyList_Size(keys_list);
//         for (Py_ssize_t i = 0; i < size; i++) {
//             PyObject *key = PyList_GetItem(keys_list, i);  // Borrowed reference
//             if (PyUnicode_Check(key)) {
//                 const char *attr_name = PyUnicode_AsUTF8(key);
//                 if (attr_name) {
//                     ereport(INFO, (errmsg("  %s", attr_name)));
//                 }
//             }
//         }
//     }

//     Py_XDECREF(keys_list);  // Cleanup
// }