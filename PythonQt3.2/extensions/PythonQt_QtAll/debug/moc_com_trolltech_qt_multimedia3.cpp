/****************************************************************************
** Meta object code from reading C++ file 'com_trolltech_qt_multimedia3.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.7.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../generated_cpp_56/com_trolltech_qt_multimedia/com_trolltech_qt_multimedia3.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'com_trolltech_qt_multimedia3.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.7.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_PythonQtWrapper_QVideoRendererControl_t {
    QByteArrayData data[13];
    char stringdata0[216];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_PythonQtWrapper_QVideoRendererControl_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_PythonQtWrapper_QVideoRendererControl_t qt_meta_stringdata_PythonQtWrapper_QVideoRendererControl = {
    {
QT_MOC_LITERAL(0, 0, 37), // "PythonQtWrapper_QVideoRendere..."
QT_MOC_LITERAL(1, 38, 25), // "new_QVideoRendererControl"
QT_MOC_LITERAL(2, 64, 22), // "QVideoRendererControl*"
QT_MOC_LITERAL(3, 87, 0), // ""
QT_MOC_LITERAL(4, 88, 6), // "parent"
QT_MOC_LITERAL(5, 95, 28), // "delete_QVideoRendererControl"
QT_MOC_LITERAL(6, 124, 3), // "obj"
QT_MOC_LITERAL(7, 128, 10), // "setSurface"
QT_MOC_LITERAL(8, 139, 16), // "theWrappedObject"
QT_MOC_LITERAL(9, 156, 22), // "QAbstractVideoSurface*"
QT_MOC_LITERAL(10, 179, 7), // "surface"
QT_MOC_LITERAL(11, 187, 15), // "py_q_setSurface"
QT_MOC_LITERAL(12, 203, 12) // "py_q_surface"

    },
    "PythonQtWrapper_QVideoRendererControl\0"
    "new_QVideoRendererControl\0"
    "QVideoRendererControl*\0\0parent\0"
    "delete_QVideoRendererControl\0obj\0"
    "setSurface\0theWrappedObject\0"
    "QAbstractVideoSurface*\0surface\0"
    "py_q_setSurface\0py_q_surface"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_PythonQtWrapper_QVideoRendererControl[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   49,    3, 0x0a /* Public */,
       1,    0,   52,    3, 0x2a /* Public | MethodCloned */,
       5,    1,   53,    3, 0x0a /* Public */,
       7,    2,   56,    3, 0x0a /* Public */,
      11,    2,   61,    3, 0x0a /* Public */,
      10,    1,   66,    3, 0x0a /* Public */,
      12,    1,   69,    3, 0x0a /* Public */,

 // slots: parameters
    0x80000000 | 2, QMetaType::QObjectStar,    4,
    0x80000000 | 2,
    QMetaType::Void, 0x80000000 | 2,    6,
    QMetaType::Void, 0x80000000 | 2, 0x80000000 | 9,    8,   10,
    QMetaType::Void, 0x80000000 | 2, 0x80000000 | 9,    8,   10,
    0x80000000 | 9, 0x80000000 | 2,    8,
    0x80000000 | 9, 0x80000000 | 2,    8,

       0        // eod
};

void PythonQtWrapper_QVideoRendererControl::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        PythonQtWrapper_QVideoRendererControl *_t = static_cast<PythonQtWrapper_QVideoRendererControl *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: { QVideoRendererControl* _r = _t->new_QVideoRendererControl((*reinterpret_cast< QObject*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QVideoRendererControl**>(_a[0]) = _r; }  break;
        case 1: { QVideoRendererControl* _r = _t->new_QVideoRendererControl();
            if (_a[0]) *reinterpret_cast< QVideoRendererControl**>(_a[0]) = _r; }  break;
        case 2: _t->delete_QVideoRendererControl((*reinterpret_cast< QVideoRendererControl*(*)>(_a[1]))); break;
        case 3: _t->setSurface((*reinterpret_cast< QVideoRendererControl*(*)>(_a[1])),(*reinterpret_cast< QAbstractVideoSurface*(*)>(_a[2]))); break;
        case 4: _t->py_q_setSurface((*reinterpret_cast< QVideoRendererControl*(*)>(_a[1])),(*reinterpret_cast< QAbstractVideoSurface*(*)>(_a[2]))); break;
        case 5: { QAbstractVideoSurface* _r = _t->surface((*reinterpret_cast< QVideoRendererControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QAbstractVideoSurface**>(_a[0]) = _r; }  break;
        case 6: { QAbstractVideoSurface* _r = _t->py_q_surface((*reinterpret_cast< QVideoRendererControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QAbstractVideoSurface**>(_a[0]) = _r; }  break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 2:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoRendererControl* >(); break;
            }
            break;
        case 3:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 1:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QAbstractVideoSurface* >(); break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoRendererControl* >(); break;
            }
            break;
        case 4:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 1:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QAbstractVideoSurface* >(); break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoRendererControl* >(); break;
            }
            break;
        case 5:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoRendererControl* >(); break;
            }
            break;
        case 6:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoRendererControl* >(); break;
            }
            break;
        }
    }
}

const QMetaObject PythonQtWrapper_QVideoRendererControl::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_PythonQtWrapper_QVideoRendererControl.data,
      qt_meta_data_PythonQtWrapper_QVideoRendererControl,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *PythonQtWrapper_QVideoRendererControl::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *PythonQtWrapper_QVideoRendererControl::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_PythonQtWrapper_QVideoRendererControl.stringdata0))
        return static_cast<void*>(const_cast< PythonQtWrapper_QVideoRendererControl*>(this));
    return QObject::qt_metacast(_clname);
}

int PythonQtWrapper_QVideoRendererControl::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    }
    return _id;
}
struct qt_meta_stringdata_PythonQtWrapper_QVideoSurfaceFormat_t {
    QByteArrayData data[59];
    char stringdata0[824];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_PythonQtWrapper_QVideoSurfaceFormat_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_PythonQtWrapper_QVideoSurfaceFormat_t qt_meta_stringdata_PythonQtWrapper_QVideoSurfaceFormat = {
    {
QT_MOC_LITERAL(0, 0, 35), // "PythonQtWrapper_QVideoSurface..."
QT_MOC_LITERAL(1, 36, 23), // "new_QVideoSurfaceFormat"
QT_MOC_LITERAL(2, 60, 20), // "QVideoSurfaceFormat*"
QT_MOC_LITERAL(3, 81, 0), // ""
QT_MOC_LITERAL(4, 82, 4), // "size"
QT_MOC_LITERAL(5, 87, 24), // "QVideoFrame::PixelFormat"
QT_MOC_LITERAL(6, 112, 11), // "pixelFormat"
QT_MOC_LITERAL(7, 124, 32), // "QAbstractVideoBuffer::HandleType"
QT_MOC_LITERAL(8, 157, 10), // "handleType"
QT_MOC_LITERAL(9, 168, 19), // "QVideoSurfaceFormat"
QT_MOC_LITERAL(10, 188, 6), // "format"
QT_MOC_LITERAL(11, 195, 26), // "delete_QVideoSurfaceFormat"
QT_MOC_LITERAL(12, 222, 3), // "obj"
QT_MOC_LITERAL(13, 226, 11), // "frameHeight"
QT_MOC_LITERAL(14, 238, 16), // "theWrappedObject"
QT_MOC_LITERAL(15, 255, 9), // "frameRate"
QT_MOC_LITERAL(16, 265, 9), // "frameSize"
QT_MOC_LITERAL(17, 275, 10), // "frameWidth"
QT_MOC_LITERAL(18, 286, 7), // "isValid"
QT_MOC_LITERAL(19, 294, 6), // "__ne__"
QT_MOC_LITERAL(20, 301, 15), // "operator_assign"
QT_MOC_LITERAL(21, 317, 6), // "__eq__"
QT_MOC_LITERAL(22, 324, 16), // "pixelAspectRatio"
QT_MOC_LITERAL(23, 341, 8), // "property"
QT_MOC_LITERAL(24, 350, 11), // "const char*"
QT_MOC_LITERAL(25, 362, 4), // "name"
QT_MOC_LITERAL(26, 367, 13), // "propertyNames"
QT_MOC_LITERAL(27, 381, 17), // "scanLineDirection"
QT_MOC_LITERAL(28, 399, 30), // "QVideoSurfaceFormat::Direction"
QT_MOC_LITERAL(29, 430, 12), // "setFrameRate"
QT_MOC_LITERAL(30, 443, 4), // "rate"
QT_MOC_LITERAL(31, 448, 12), // "setFrameSize"
QT_MOC_LITERAL(32, 461, 5), // "width"
QT_MOC_LITERAL(33, 467, 6), // "height"
QT_MOC_LITERAL(34, 474, 19), // "setPixelAspectRatio"
QT_MOC_LITERAL(35, 494, 5), // "ratio"
QT_MOC_LITERAL(36, 500, 11), // "setProperty"
QT_MOC_LITERAL(37, 512, 5), // "value"
QT_MOC_LITERAL(38, 518, 20), // "setScanLineDirection"
QT_MOC_LITERAL(39, 539, 9), // "direction"
QT_MOC_LITERAL(40, 549, 11), // "setViewport"
QT_MOC_LITERAL(41, 561, 8), // "viewport"
QT_MOC_LITERAL(42, 570, 18), // "setYCbCrColorSpace"
QT_MOC_LITERAL(43, 589, 36), // "QVideoSurfaceFormat::YCbCrCol..."
QT_MOC_LITERAL(44, 626, 10), // "colorSpace"
QT_MOC_LITERAL(45, 637, 8), // "sizeHint"
QT_MOC_LITERAL(46, 646, 15), // "yCbCrColorSpace"
QT_MOC_LITERAL(47, 662, 11), // "py_toString"
QT_MOC_LITERAL(48, 674, 9), // "Direction"
QT_MOC_LITERAL(49, 684, 11), // "TopToBottom"
QT_MOC_LITERAL(50, 696, 11), // "BottomToTop"
QT_MOC_LITERAL(51, 708, 15), // "YCbCrColorSpace"
QT_MOC_LITERAL(52, 724, 15), // "YCbCr_Undefined"
QT_MOC_LITERAL(53, 740, 11), // "YCbCr_BT601"
QT_MOC_LITERAL(54, 752, 11), // "YCbCr_BT709"
QT_MOC_LITERAL(55, 764, 14), // "YCbCr_xvYCC601"
QT_MOC_LITERAL(56, 779, 14), // "YCbCr_xvYCC709"
QT_MOC_LITERAL(57, 794, 10), // "YCbCr_JPEG"
QT_MOC_LITERAL(58, 805, 18) // "YCbCr_CustomMatrix"

    },
    "PythonQtWrapper_QVideoSurfaceFormat\0"
    "new_QVideoSurfaceFormat\0QVideoSurfaceFormat*\0"
    "\0size\0QVideoFrame::PixelFormat\0"
    "pixelFormat\0QAbstractVideoBuffer::HandleType\0"
    "handleType\0QVideoSurfaceFormat\0format\0"
    "delete_QVideoSurfaceFormat\0obj\0"
    "frameHeight\0theWrappedObject\0frameRate\0"
    "frameSize\0frameWidth\0isValid\0__ne__\0"
    "operator_assign\0__eq__\0pixelAspectRatio\0"
    "property\0const char*\0name\0propertyNames\0"
    "scanLineDirection\0QVideoSurfaceFormat::Direction\0"
    "setFrameRate\0rate\0setFrameSize\0width\0"
    "height\0setPixelAspectRatio\0ratio\0"
    "setProperty\0value\0setScanLineDirection\0"
    "direction\0setViewport\0viewport\0"
    "setYCbCrColorSpace\0"
    "QVideoSurfaceFormat::YCbCrColorSpace\0"
    "colorSpace\0sizeHint\0yCbCrColorSpace\0"
    "py_toString\0Direction\0TopToBottom\0"
    "BottomToTop\0YCbCrColorSpace\0YCbCr_Undefined\0"
    "YCbCr_BT601\0YCbCr_BT709\0YCbCr_xvYCC601\0"
    "YCbCr_xvYCC709\0YCbCr_JPEG\0YCbCr_CustomMatrix"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_PythonQtWrapper_QVideoSurfaceFormat[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      32,   14, // methods
       0,    0, // properties
       2,  306, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,  174,    3, 0x0a /* Public */,
       1,    3,  175,    3, 0x0a /* Public */,
       1,    2,  182,    3, 0x2a /* Public | MethodCloned */,
       1,    1,  187,    3, 0x0a /* Public */,
      11,    1,  190,    3, 0x0a /* Public */,
      13,    1,  193,    3, 0x0a /* Public */,
      15,    1,  196,    3, 0x0a /* Public */,
      16,    1,  199,    3, 0x0a /* Public */,
      17,    1,  202,    3, 0x0a /* Public */,
       8,    1,  205,    3, 0x0a /* Public */,
      18,    1,  208,    3, 0x0a /* Public */,
      19,    2,  211,    3, 0x0a /* Public */,
      20,    2,  216,    3, 0x0a /* Public */,
      21,    2,  221,    3, 0x0a /* Public */,
      22,    1,  226,    3, 0x0a /* Public */,
       6,    1,  229,    3, 0x0a /* Public */,
      23,    2,  232,    3, 0x0a /* Public */,
      26,    1,  237,    3, 0x0a /* Public */,
      27,    1,  240,    3, 0x0a /* Public */,
      29,    2,  243,    3, 0x0a /* Public */,
      31,    2,  248,    3, 0x0a /* Public */,
      31,    3,  253,    3, 0x0a /* Public */,
      34,    2,  260,    3, 0x0a /* Public */,
      34,    3,  265,    3, 0x0a /* Public */,
      36,    3,  272,    3, 0x0a /* Public */,
      38,    2,  279,    3, 0x0a /* Public */,
      40,    2,  284,    3, 0x0a /* Public */,
      42,    2,  289,    3, 0x0a /* Public */,
      45,    1,  294,    3, 0x0a /* Public */,
      41,    1,  297,    3, 0x0a /* Public */,
      46,    1,  300,    3, 0x0a /* Public */,
      47,    1,  303,    3, 0x0a /* Public */,

 // slots: parameters
    0x80000000 | 2,
    0x80000000 | 2, QMetaType::QSize, 0x80000000 | 5, 0x80000000 | 7,    4,    6,    8,
    0x80000000 | 2, QMetaType::QSize, 0x80000000 | 5,    4,    6,
    0x80000000 | 2, 0x80000000 | 9,   10,
    QMetaType::Void, 0x80000000 | 2,   12,
    QMetaType::Int, 0x80000000 | 2,   14,
    QMetaType::QReal, 0x80000000 | 2,   14,
    QMetaType::QSize, 0x80000000 | 2,   14,
    QMetaType::Int, 0x80000000 | 2,   14,
    0x80000000 | 7, 0x80000000 | 2,   14,
    QMetaType::Bool, 0x80000000 | 2,   14,
    QMetaType::Bool, 0x80000000 | 2, 0x80000000 | 9,   14,   10,
    0x80000000 | 2, 0x80000000 | 2, 0x80000000 | 9,   14,   10,
    QMetaType::Bool, 0x80000000 | 2, 0x80000000 | 9,   14,   10,
    QMetaType::QSize, 0x80000000 | 2,   14,
    0x80000000 | 5, 0x80000000 | 2,   14,
    QMetaType::QVariant, 0x80000000 | 2, 0x80000000 | 24,   14,   25,
    QMetaType::QByteArrayList, 0x80000000 | 2,   14,
    0x80000000 | 28, 0x80000000 | 2,   14,
    QMetaType::Void, 0x80000000 | 2, QMetaType::QReal,   14,   30,
    QMetaType::Void, 0x80000000 | 2, QMetaType::QSize,   14,    4,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Int, QMetaType::Int,   14,   32,   33,
    QMetaType::Void, 0x80000000 | 2, QMetaType::QSize,   14,   35,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Int, QMetaType::Int,   14,   32,   33,
    QMetaType::Void, 0x80000000 | 2, 0x80000000 | 24, QMetaType::QVariant,   14,   25,   37,
    QMetaType::Void, 0x80000000 | 2, 0x80000000 | 28,   14,   39,
    QMetaType::Void, 0x80000000 | 2, QMetaType::QRect,   14,   41,
    QMetaType::Void, 0x80000000 | 2, 0x80000000 | 43,   14,   44,
    QMetaType::QSize, 0x80000000 | 2,   14,
    QMetaType::QRect, 0x80000000 | 2,   14,
    0x80000000 | 43, 0x80000000 | 2,   14,
    QMetaType::QString, 0x80000000 | 2,    3,

 // enums: name, flags, count, data
      48, 0x0,    2,  314,
      51, 0x0,    7,  318,

 // enum data: key, value
      49, uint(PythonQtWrapper_QVideoSurfaceFormat::TopToBottom),
      50, uint(PythonQtWrapper_QVideoSurfaceFormat::BottomToTop),
      52, uint(PythonQtWrapper_QVideoSurfaceFormat::YCbCr_Undefined),
      53, uint(PythonQtWrapper_QVideoSurfaceFormat::YCbCr_BT601),
      54, uint(PythonQtWrapper_QVideoSurfaceFormat::YCbCr_BT709),
      55, uint(PythonQtWrapper_QVideoSurfaceFormat::YCbCr_xvYCC601),
      56, uint(PythonQtWrapper_QVideoSurfaceFormat::YCbCr_xvYCC709),
      57, uint(PythonQtWrapper_QVideoSurfaceFormat::YCbCr_JPEG),
      58, uint(PythonQtWrapper_QVideoSurfaceFormat::YCbCr_CustomMatrix),

       0        // eod
};

void PythonQtWrapper_QVideoSurfaceFormat::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        PythonQtWrapper_QVideoSurfaceFormat *_t = static_cast<PythonQtWrapper_QVideoSurfaceFormat *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: { QVideoSurfaceFormat* _r = _t->new_QVideoSurfaceFormat();
            if (_a[0]) *reinterpret_cast< QVideoSurfaceFormat**>(_a[0]) = _r; }  break;
        case 1: { QVideoSurfaceFormat* _r = _t->new_QVideoSurfaceFormat((*reinterpret_cast< const QSize(*)>(_a[1])),(*reinterpret_cast< QVideoFrame::PixelFormat(*)>(_a[2])),(*reinterpret_cast< QAbstractVideoBuffer::HandleType(*)>(_a[3])));
            if (_a[0]) *reinterpret_cast< QVideoSurfaceFormat**>(_a[0]) = _r; }  break;
        case 2: { QVideoSurfaceFormat* _r = _t->new_QVideoSurfaceFormat((*reinterpret_cast< const QSize(*)>(_a[1])),(*reinterpret_cast< QVideoFrame::PixelFormat(*)>(_a[2])));
            if (_a[0]) *reinterpret_cast< QVideoSurfaceFormat**>(_a[0]) = _r; }  break;
        case 3: { QVideoSurfaceFormat* _r = _t->new_QVideoSurfaceFormat((*reinterpret_cast< const QVideoSurfaceFormat(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QVideoSurfaceFormat**>(_a[0]) = _r; }  break;
        case 4: _t->delete_QVideoSurfaceFormat((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1]))); break;
        case 5: { int _r = _t->frameHeight((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 6: { qreal _r = _t->frameRate((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< qreal*>(_a[0]) = _r; }  break;
        case 7: { QSize _r = _t->frameSize((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QSize*>(_a[0]) = _r; }  break;
        case 8: { int _r = _t->frameWidth((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 9: { QAbstractVideoBuffer::HandleType _r = _t->handleType((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QAbstractVideoBuffer::HandleType*>(_a[0]) = _r; }  break;
        case 10: { bool _r = _t->isValid((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        case 11: { bool _r = _t->__ne__((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])),(*reinterpret_cast< const QVideoSurfaceFormat(*)>(_a[2])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        case 12: { QVideoSurfaceFormat* _r = _t->operator_assign((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])),(*reinterpret_cast< const QVideoSurfaceFormat(*)>(_a[2])));
            if (_a[0]) *reinterpret_cast< QVideoSurfaceFormat**>(_a[0]) = _r; }  break;
        case 13: { bool _r = _t->__eq__((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])),(*reinterpret_cast< const QVideoSurfaceFormat(*)>(_a[2])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        case 14: { QSize _r = _t->pixelAspectRatio((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QSize*>(_a[0]) = _r; }  break;
        case 15: { QVideoFrame::PixelFormat _r = _t->pixelFormat((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QVideoFrame::PixelFormat*>(_a[0]) = _r; }  break;
        case 16: { QVariant _r = _t->property((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])),(*reinterpret_cast< const char*(*)>(_a[2])));
            if (_a[0]) *reinterpret_cast< QVariant*>(_a[0]) = _r; }  break;
        case 17: { QList<QByteArray> _r = _t->propertyNames((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QList<QByteArray>*>(_a[0]) = _r; }  break;
        case 18: { QVideoSurfaceFormat::Direction _r = _t->scanLineDirection((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QVideoSurfaceFormat::Direction*>(_a[0]) = _r; }  break;
        case 19: _t->setFrameRate((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])),(*reinterpret_cast< qreal(*)>(_a[2]))); break;
        case 20: _t->setFrameSize((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])),(*reinterpret_cast< const QSize(*)>(_a[2]))); break;
        case 21: _t->setFrameSize((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 22: _t->setPixelAspectRatio((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])),(*reinterpret_cast< const QSize(*)>(_a[2]))); break;
        case 23: _t->setPixelAspectRatio((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 24: _t->setProperty((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])),(*reinterpret_cast< const char*(*)>(_a[2])),(*reinterpret_cast< const QVariant(*)>(_a[3]))); break;
        case 25: _t->setScanLineDirection((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])),(*reinterpret_cast< QVideoSurfaceFormat::Direction(*)>(_a[2]))); break;
        case 26: _t->setViewport((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])),(*reinterpret_cast< const QRect(*)>(_a[2]))); break;
        case 27: _t->setYCbCrColorSpace((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])),(*reinterpret_cast< QVideoSurfaceFormat::YCbCrColorSpace(*)>(_a[2]))); break;
        case 28: { QSize _r = _t->sizeHint((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QSize*>(_a[0]) = _r; }  break;
        case 29: { QRect _r = _t->viewport((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QRect*>(_a[0]) = _r; }  break;
        case 30: { QVideoSurfaceFormat::YCbCrColorSpace _r = _t->yCbCrColorSpace((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QVideoSurfaceFormat::YCbCrColorSpace*>(_a[0]) = _r; }  break;
        case 31: { QString _r = _t->py_toString((*reinterpret_cast< QVideoSurfaceFormat*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QString*>(_a[0]) = _r; }  break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 1:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 2:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QAbstractVideoBuffer::HandleType >(); break;
            case 1:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoFrame::PixelFormat >(); break;
            }
            break;
        case 2:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 1:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoFrame::PixelFormat >(); break;
            }
            break;
        case 3:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoSurfaceFormat >(); break;
            }
            break;
        case 11:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 1:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoSurfaceFormat >(); break;
            }
            break;
        case 12:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 1:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoSurfaceFormat >(); break;
            }
            break;
        case 13:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 1:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoSurfaceFormat >(); break;
            }
            break;
        case 25:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 1:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoSurfaceFormat::Direction >(); break;
            }
            break;
        case 27:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 1:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoSurfaceFormat::YCbCrColorSpace >(); break;
            }
            break;
        }
    }
}

const QMetaObject PythonQtWrapper_QVideoSurfaceFormat::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_PythonQtWrapper_QVideoSurfaceFormat.data,
      qt_meta_data_PythonQtWrapper_QVideoSurfaceFormat,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *PythonQtWrapper_QVideoSurfaceFormat::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *PythonQtWrapper_QVideoSurfaceFormat::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_PythonQtWrapper_QVideoSurfaceFormat.stringdata0))
        return static_cast<void*>(const_cast< PythonQtWrapper_QVideoSurfaceFormat*>(this));
    return QObject::qt_metacast(_clname);
}

int PythonQtWrapper_QVideoSurfaceFormat::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 32)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 32;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 32)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 32;
    }
    return _id;
}
struct qt_meta_stringdata_PythonQtWrapper_QVideoWidget_t {
    QByteArrayData data[33];
    char stringdata0[422];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_PythonQtWrapper_QVideoWidget_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_PythonQtWrapper_QVideoWidget_t qt_meta_stringdata_PythonQtWrapper_QVideoWidget = {
    {
QT_MOC_LITERAL(0, 0, 28), // "PythonQtWrapper_QVideoWidget"
QT_MOC_LITERAL(1, 29, 16), // "new_QVideoWidget"
QT_MOC_LITERAL(2, 46, 13), // "QVideoWidget*"
QT_MOC_LITERAL(3, 60, 0), // ""
QT_MOC_LITERAL(4, 61, 8), // "QWidget*"
QT_MOC_LITERAL(5, 70, 6), // "parent"
QT_MOC_LITERAL(6, 77, 19), // "delete_QVideoWidget"
QT_MOC_LITERAL(7, 97, 3), // "obj"
QT_MOC_LITERAL(8, 101, 15), // "aspectRatioMode"
QT_MOC_LITERAL(9, 117, 19), // "Qt::AspectRatioMode"
QT_MOC_LITERAL(10, 137, 16), // "theWrappedObject"
QT_MOC_LITERAL(11, 154, 10), // "brightness"
QT_MOC_LITERAL(12, 165, 8), // "contrast"
QT_MOC_LITERAL(13, 174, 10), // "py_q_event"
QT_MOC_LITERAL(14, 185, 7), // "QEvent*"
QT_MOC_LITERAL(15, 193, 5), // "event"
QT_MOC_LITERAL(16, 199, 14), // "py_q_hideEvent"
QT_MOC_LITERAL(17, 214, 11), // "QHideEvent*"
QT_MOC_LITERAL(18, 226, 3), // "hue"
QT_MOC_LITERAL(19, 230, 16), // "py_q_mediaObject"
QT_MOC_LITERAL(20, 247, 13), // "QMediaObject*"
QT_MOC_LITERAL(21, 261, 14), // "py_q_moveEvent"
QT_MOC_LITERAL(22, 276, 11), // "QMoveEvent*"
QT_MOC_LITERAL(23, 288, 15), // "py_q_paintEvent"
QT_MOC_LITERAL(24, 304, 12), // "QPaintEvent*"
QT_MOC_LITERAL(25, 317, 16), // "py_q_resizeEvent"
QT_MOC_LITERAL(26, 334, 13), // "QResizeEvent*"
QT_MOC_LITERAL(27, 348, 10), // "saturation"
QT_MOC_LITERAL(28, 359, 19), // "py_q_setMediaObject"
QT_MOC_LITERAL(29, 379, 6), // "object"
QT_MOC_LITERAL(30, 386, 14), // "py_q_showEvent"
QT_MOC_LITERAL(31, 401, 11), // "QShowEvent*"
QT_MOC_LITERAL(32, 413, 8) // "sizeHint"

    },
    "PythonQtWrapper_QVideoWidget\0"
    "new_QVideoWidget\0QVideoWidget*\0\0"
    "QWidget*\0parent\0delete_QVideoWidget\0"
    "obj\0aspectRatioMode\0Qt::AspectRatioMode\0"
    "theWrappedObject\0brightness\0contrast\0"
    "py_q_event\0QEvent*\0event\0py_q_hideEvent\0"
    "QHideEvent*\0hue\0py_q_mediaObject\0"
    "QMediaObject*\0py_q_moveEvent\0QMoveEvent*\0"
    "py_q_paintEvent\0QPaintEvent*\0"
    "py_q_resizeEvent\0QResizeEvent*\0"
    "saturation\0py_q_setMediaObject\0object\0"
    "py_q_showEvent\0QShowEvent*\0sizeHint"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_PythonQtWrapper_QVideoWidget[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      17,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   99,    3, 0x0a /* Public */,
       1,    0,  102,    3, 0x2a /* Public | MethodCloned */,
       6,    1,  103,    3, 0x0a /* Public */,
       8,    1,  106,    3, 0x0a /* Public */,
      11,    1,  109,    3, 0x0a /* Public */,
      12,    1,  112,    3, 0x0a /* Public */,
      13,    2,  115,    3, 0x0a /* Public */,
      16,    2,  120,    3, 0x0a /* Public */,
      18,    1,  125,    3, 0x0a /* Public */,
      19,    1,  128,    3, 0x0a /* Public */,
      21,    2,  131,    3, 0x0a /* Public */,
      23,    2,  136,    3, 0x0a /* Public */,
      25,    2,  141,    3, 0x0a /* Public */,
      27,    1,  146,    3, 0x0a /* Public */,
      28,    2,  149,    3, 0x0a /* Public */,
      30,    2,  154,    3, 0x0a /* Public */,
      32,    1,  159,    3, 0x0a /* Public */,

 // slots: parameters
    0x80000000 | 2, 0x80000000 | 4,    5,
    0x80000000 | 2,
    QMetaType::Void, 0x80000000 | 2,    7,
    0x80000000 | 9, 0x80000000 | 2,   10,
    QMetaType::Int, 0x80000000 | 2,   10,
    QMetaType::Int, 0x80000000 | 2,   10,
    QMetaType::Bool, 0x80000000 | 2, 0x80000000 | 14,   10,   15,
    QMetaType::Void, 0x80000000 | 2, 0x80000000 | 17,   10,   15,
    QMetaType::Int, 0x80000000 | 2,   10,
    0x80000000 | 20, 0x80000000 | 2,   10,
    QMetaType::Void, 0x80000000 | 2, 0x80000000 | 22,   10,   15,
    QMetaType::Void, 0x80000000 | 2, 0x80000000 | 24,   10,   15,
    QMetaType::Void, 0x80000000 | 2, 0x80000000 | 26,   10,   15,
    QMetaType::Int, 0x80000000 | 2,   10,
    QMetaType::Bool, 0x80000000 | 2, 0x80000000 | 20,   10,   29,
    QMetaType::Void, 0x80000000 | 2, 0x80000000 | 31,   10,   15,
    QMetaType::QSize, 0x80000000 | 2,   10,

       0        // eod
};

void PythonQtWrapper_QVideoWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        PythonQtWrapper_QVideoWidget *_t = static_cast<PythonQtWrapper_QVideoWidget *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: { QVideoWidget* _r = _t->new_QVideoWidget((*reinterpret_cast< QWidget*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QVideoWidget**>(_a[0]) = _r; }  break;
        case 1: { QVideoWidget* _r = _t->new_QVideoWidget();
            if (_a[0]) *reinterpret_cast< QVideoWidget**>(_a[0]) = _r; }  break;
        case 2: _t->delete_QVideoWidget((*reinterpret_cast< QVideoWidget*(*)>(_a[1]))); break;
        case 3: { Qt::AspectRatioMode _r = _t->aspectRatioMode((*reinterpret_cast< QVideoWidget*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< Qt::AspectRatioMode*>(_a[0]) = _r; }  break;
        case 4: { int _r = _t->brightness((*reinterpret_cast< QVideoWidget*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 5: { int _r = _t->contrast((*reinterpret_cast< QVideoWidget*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 6: { bool _r = _t->py_q_event((*reinterpret_cast< QVideoWidget*(*)>(_a[1])),(*reinterpret_cast< QEvent*(*)>(_a[2])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        case 7: _t->py_q_hideEvent((*reinterpret_cast< QVideoWidget*(*)>(_a[1])),(*reinterpret_cast< QHideEvent*(*)>(_a[2]))); break;
        case 8: { int _r = _t->hue((*reinterpret_cast< QVideoWidget*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 9: { QMediaObject* _r = _t->py_q_mediaObject((*reinterpret_cast< QVideoWidget*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QMediaObject**>(_a[0]) = _r; }  break;
        case 10: _t->py_q_moveEvent((*reinterpret_cast< QVideoWidget*(*)>(_a[1])),(*reinterpret_cast< QMoveEvent*(*)>(_a[2]))); break;
        case 11: _t->py_q_paintEvent((*reinterpret_cast< QVideoWidget*(*)>(_a[1])),(*reinterpret_cast< QPaintEvent*(*)>(_a[2]))); break;
        case 12: _t->py_q_resizeEvent((*reinterpret_cast< QVideoWidget*(*)>(_a[1])),(*reinterpret_cast< QResizeEvent*(*)>(_a[2]))); break;
        case 13: { int _r = _t->saturation((*reinterpret_cast< QVideoWidget*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 14: { bool _r = _t->py_q_setMediaObject((*reinterpret_cast< QVideoWidget*(*)>(_a[1])),(*reinterpret_cast< QMediaObject*(*)>(_a[2])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        case 15: _t->py_q_showEvent((*reinterpret_cast< QVideoWidget*(*)>(_a[1])),(*reinterpret_cast< QShowEvent*(*)>(_a[2]))); break;
        case 16: { QSize _r = _t->sizeHint((*reinterpret_cast< QVideoWidget*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QSize*>(_a[0]) = _r; }  break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 0:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QWidget* >(); break;
            }
            break;
        case 2:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidget* >(); break;
            }
            break;
        case 3:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidget* >(); break;
            }
            break;
        case 4:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidget* >(); break;
            }
            break;
        case 5:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidget* >(); break;
            }
            break;
        case 6:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidget* >(); break;
            }
            break;
        case 7:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidget* >(); break;
            }
            break;
        case 8:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidget* >(); break;
            }
            break;
        case 9:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidget* >(); break;
            }
            break;
        case 10:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidget* >(); break;
            }
            break;
        case 11:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidget* >(); break;
            }
            break;
        case 12:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidget* >(); break;
            }
            break;
        case 13:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidget* >(); break;
            }
            break;
        case 14:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 1:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QMediaObject* >(); break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidget* >(); break;
            }
            break;
        case 15:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidget* >(); break;
            }
            break;
        case 16:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidget* >(); break;
            }
            break;
        }
    }
}

const QMetaObject PythonQtWrapper_QVideoWidget::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_PythonQtWrapper_QVideoWidget.data,
      qt_meta_data_PythonQtWrapper_QVideoWidget,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *PythonQtWrapper_QVideoWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *PythonQtWrapper_QVideoWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_PythonQtWrapper_QVideoWidget.stringdata0))
        return static_cast<void*>(const_cast< PythonQtWrapper_QVideoWidget*>(this));
    return QObject::qt_metacast(_clname);
}

int PythonQtWrapper_QVideoWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 17)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 17;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 17)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 17;
    }
    return _id;
}
struct qt_meta_stringdata_PythonQtWrapper_QVideoWidgetControl_t {
    QByteArrayData data[38];
    char stringdata0[559];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_PythonQtWrapper_QVideoWidgetControl_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_PythonQtWrapper_QVideoWidgetControl_t qt_meta_stringdata_PythonQtWrapper_QVideoWidgetControl = {
    {
QT_MOC_LITERAL(0, 0, 35), // "PythonQtWrapper_QVideoWidgetC..."
QT_MOC_LITERAL(1, 36, 23), // "new_QVideoWidgetControl"
QT_MOC_LITERAL(2, 60, 20), // "QVideoWidgetControl*"
QT_MOC_LITERAL(3, 81, 0), // ""
QT_MOC_LITERAL(4, 82, 6), // "parent"
QT_MOC_LITERAL(5, 89, 26), // "delete_QVideoWidgetControl"
QT_MOC_LITERAL(6, 116, 3), // "obj"
QT_MOC_LITERAL(7, 120, 15), // "aspectRatioMode"
QT_MOC_LITERAL(8, 136, 19), // "Qt::AspectRatioMode"
QT_MOC_LITERAL(9, 156, 16), // "theWrappedObject"
QT_MOC_LITERAL(10, 173, 20), // "py_q_aspectRatioMode"
QT_MOC_LITERAL(11, 194, 10), // "brightness"
QT_MOC_LITERAL(12, 205, 15), // "py_q_brightness"
QT_MOC_LITERAL(13, 221, 8), // "contrast"
QT_MOC_LITERAL(14, 230, 13), // "py_q_contrast"
QT_MOC_LITERAL(15, 244, 3), // "hue"
QT_MOC_LITERAL(16, 248, 8), // "py_q_hue"
QT_MOC_LITERAL(17, 257, 12), // "isFullScreen"
QT_MOC_LITERAL(18, 270, 17), // "py_q_isFullScreen"
QT_MOC_LITERAL(19, 288, 10), // "saturation"
QT_MOC_LITERAL(20, 299, 15), // "py_q_saturation"
QT_MOC_LITERAL(21, 315, 18), // "setAspectRatioMode"
QT_MOC_LITERAL(22, 334, 4), // "mode"
QT_MOC_LITERAL(23, 339, 23), // "py_q_setAspectRatioMode"
QT_MOC_LITERAL(24, 363, 13), // "setBrightness"
QT_MOC_LITERAL(25, 377, 18), // "py_q_setBrightness"
QT_MOC_LITERAL(26, 396, 11), // "setContrast"
QT_MOC_LITERAL(27, 408, 16), // "py_q_setContrast"
QT_MOC_LITERAL(28, 425, 13), // "setFullScreen"
QT_MOC_LITERAL(29, 439, 10), // "fullScreen"
QT_MOC_LITERAL(30, 450, 18), // "py_q_setFullScreen"
QT_MOC_LITERAL(31, 469, 6), // "setHue"
QT_MOC_LITERAL(32, 476, 11), // "py_q_setHue"
QT_MOC_LITERAL(33, 488, 13), // "setSaturation"
QT_MOC_LITERAL(34, 502, 18), // "py_q_setSaturation"
QT_MOC_LITERAL(35, 521, 11), // "videoWidget"
QT_MOC_LITERAL(36, 533, 8), // "QWidget*"
QT_MOC_LITERAL(37, 542, 16) // "py_q_videoWidget"

    },
    "PythonQtWrapper_QVideoWidgetControl\0"
    "new_QVideoWidgetControl\0QVideoWidgetControl*\0"
    "\0parent\0delete_QVideoWidgetControl\0"
    "obj\0aspectRatioMode\0Qt::AspectRatioMode\0"
    "theWrappedObject\0py_q_aspectRatioMode\0"
    "brightness\0py_q_brightness\0contrast\0"
    "py_q_contrast\0hue\0py_q_hue\0isFullScreen\0"
    "py_q_isFullScreen\0saturation\0"
    "py_q_saturation\0setAspectRatioMode\0"
    "mode\0py_q_setAspectRatioMode\0setBrightness\0"
    "py_q_setBrightness\0setContrast\0"
    "py_q_setContrast\0setFullScreen\0"
    "fullScreen\0py_q_setFullScreen\0setHue\0"
    "py_q_setHue\0setSaturation\0py_q_setSaturation\0"
    "videoWidget\0QWidget*\0py_q_videoWidget"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_PythonQtWrapper_QVideoWidgetControl[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      29,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,  159,    3, 0x0a /* Public */,
       1,    0,  162,    3, 0x2a /* Public | MethodCloned */,
       5,    1,  163,    3, 0x0a /* Public */,
       7,    1,  166,    3, 0x0a /* Public */,
      10,    1,  169,    3, 0x0a /* Public */,
      11,    1,  172,    3, 0x0a /* Public */,
      12,    1,  175,    3, 0x0a /* Public */,
      13,    1,  178,    3, 0x0a /* Public */,
      14,    1,  181,    3, 0x0a /* Public */,
      15,    1,  184,    3, 0x0a /* Public */,
      16,    1,  187,    3, 0x0a /* Public */,
      17,    1,  190,    3, 0x0a /* Public */,
      18,    1,  193,    3, 0x0a /* Public */,
      19,    1,  196,    3, 0x0a /* Public */,
      20,    1,  199,    3, 0x0a /* Public */,
      21,    2,  202,    3, 0x0a /* Public */,
      23,    2,  207,    3, 0x0a /* Public */,
      24,    2,  212,    3, 0x0a /* Public */,
      25,    2,  217,    3, 0x0a /* Public */,
      26,    2,  222,    3, 0x0a /* Public */,
      27,    2,  227,    3, 0x0a /* Public */,
      28,    2,  232,    3, 0x0a /* Public */,
      30,    2,  237,    3, 0x0a /* Public */,
      31,    2,  242,    3, 0x0a /* Public */,
      32,    2,  247,    3, 0x0a /* Public */,
      33,    2,  252,    3, 0x0a /* Public */,
      34,    2,  257,    3, 0x0a /* Public */,
      35,    1,  262,    3, 0x0a /* Public */,
      37,    1,  265,    3, 0x0a /* Public */,

 // slots: parameters
    0x80000000 | 2, QMetaType::QObjectStar,    4,
    0x80000000 | 2,
    QMetaType::Void, 0x80000000 | 2,    6,
    0x80000000 | 8, 0x80000000 | 2,    9,
    0x80000000 | 8, 0x80000000 | 2,    9,
    QMetaType::Int, 0x80000000 | 2,    9,
    QMetaType::Int, 0x80000000 | 2,    9,
    QMetaType::Int, 0x80000000 | 2,    9,
    QMetaType::Int, 0x80000000 | 2,    9,
    QMetaType::Int, 0x80000000 | 2,    9,
    QMetaType::Int, 0x80000000 | 2,    9,
    QMetaType::Bool, 0x80000000 | 2,    9,
    QMetaType::Bool, 0x80000000 | 2,    9,
    QMetaType::Int, 0x80000000 | 2,    9,
    QMetaType::Int, 0x80000000 | 2,    9,
    QMetaType::Void, 0x80000000 | 2, 0x80000000 | 8,    9,   22,
    QMetaType::Void, 0x80000000 | 2, 0x80000000 | 8,    9,   22,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Int,    9,   11,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Int,    9,   11,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Int,    9,   13,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Int,    9,   13,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Bool,    9,   29,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Bool,    9,   29,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Int,    9,   15,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Int,    9,   15,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Int,    9,   19,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Int,    9,   19,
    0x80000000 | 36, 0x80000000 | 2,    9,
    0x80000000 | 36, 0x80000000 | 2,    9,

       0        // eod
};

void PythonQtWrapper_QVideoWidgetControl::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        PythonQtWrapper_QVideoWidgetControl *_t = static_cast<PythonQtWrapper_QVideoWidgetControl *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: { QVideoWidgetControl* _r = _t->new_QVideoWidgetControl((*reinterpret_cast< QObject*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QVideoWidgetControl**>(_a[0]) = _r; }  break;
        case 1: { QVideoWidgetControl* _r = _t->new_QVideoWidgetControl();
            if (_a[0]) *reinterpret_cast< QVideoWidgetControl**>(_a[0]) = _r; }  break;
        case 2: _t->delete_QVideoWidgetControl((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1]))); break;
        case 3: { Qt::AspectRatioMode _r = _t->aspectRatioMode((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< Qt::AspectRatioMode*>(_a[0]) = _r; }  break;
        case 4: { Qt::AspectRatioMode _r = _t->py_q_aspectRatioMode((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< Qt::AspectRatioMode*>(_a[0]) = _r; }  break;
        case 5: { int _r = _t->brightness((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 6: { int _r = _t->py_q_brightness((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 7: { int _r = _t->contrast((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 8: { int _r = _t->py_q_contrast((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 9: { int _r = _t->hue((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 10: { int _r = _t->py_q_hue((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 11: { bool _r = _t->isFullScreen((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        case 12: { bool _r = _t->py_q_isFullScreen((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        case 13: { int _r = _t->saturation((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 14: { int _r = _t->py_q_saturation((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 15: _t->setAspectRatioMode((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])),(*reinterpret_cast< Qt::AspectRatioMode(*)>(_a[2]))); break;
        case 16: _t->py_q_setAspectRatioMode((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])),(*reinterpret_cast< Qt::AspectRatioMode(*)>(_a[2]))); break;
        case 17: _t->setBrightness((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 18: _t->py_q_setBrightness((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 19: _t->setContrast((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 20: _t->py_q_setContrast((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 21: _t->setFullScreen((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])),(*reinterpret_cast< bool(*)>(_a[2]))); break;
        case 22: _t->py_q_setFullScreen((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])),(*reinterpret_cast< bool(*)>(_a[2]))); break;
        case 23: _t->setHue((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 24: _t->py_q_setHue((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 25: _t->setSaturation((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 26: _t->py_q_setSaturation((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 27: { QWidget* _r = _t->videoWidget((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QWidget**>(_a[0]) = _r; }  break;
        case 28: { QWidget* _r = _t->py_q_videoWidget((*reinterpret_cast< QVideoWidgetControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QWidget**>(_a[0]) = _r; }  break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 2:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 3:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 4:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 5:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 6:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 7:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 8:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 9:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 10:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 11:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 12:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 13:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 14:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 15:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 16:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 17:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 18:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 19:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 20:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 21:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 22:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 23:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 24:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 25:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 26:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 27:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        case 28:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWidgetControl* >(); break;
            }
            break;
        }
    }
}

const QMetaObject PythonQtWrapper_QVideoWidgetControl::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_PythonQtWrapper_QVideoWidgetControl.data,
      qt_meta_data_PythonQtWrapper_QVideoWidgetControl,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *PythonQtWrapper_QVideoWidgetControl::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *PythonQtWrapper_QVideoWidgetControl::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_PythonQtWrapper_QVideoWidgetControl.stringdata0))
        return static_cast<void*>(const_cast< PythonQtWrapper_QVideoWidgetControl*>(this));
    return QObject::qt_metacast(_clname);
}

int PythonQtWrapper_QVideoWidgetControl::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 29)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 29;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 29)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 29;
    }
    return _id;
}
struct qt_meta_stringdata_PythonQtWrapper_QVideoWindowControl_t {
    QByteArrayData data[50];
    char stringdata0[685];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_PythonQtWrapper_QVideoWindowControl_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_PythonQtWrapper_QVideoWindowControl_t qt_meta_stringdata_PythonQtWrapper_QVideoWindowControl = {
    {
QT_MOC_LITERAL(0, 0, 35), // "PythonQtWrapper_QVideoWindowC..."
QT_MOC_LITERAL(1, 36, 23), // "new_QVideoWindowControl"
QT_MOC_LITERAL(2, 60, 20), // "QVideoWindowControl*"
QT_MOC_LITERAL(3, 81, 0), // ""
QT_MOC_LITERAL(4, 82, 6), // "parent"
QT_MOC_LITERAL(5, 89, 26), // "delete_QVideoWindowControl"
QT_MOC_LITERAL(6, 116, 3), // "obj"
QT_MOC_LITERAL(7, 120, 15), // "aspectRatioMode"
QT_MOC_LITERAL(8, 136, 19), // "Qt::AspectRatioMode"
QT_MOC_LITERAL(9, 156, 16), // "theWrappedObject"
QT_MOC_LITERAL(10, 173, 20), // "py_q_aspectRatioMode"
QT_MOC_LITERAL(11, 194, 10), // "brightness"
QT_MOC_LITERAL(12, 205, 15), // "py_q_brightness"
QT_MOC_LITERAL(13, 221, 8), // "contrast"
QT_MOC_LITERAL(14, 230, 13), // "py_q_contrast"
QT_MOC_LITERAL(15, 244, 11), // "displayRect"
QT_MOC_LITERAL(16, 256, 16), // "py_q_displayRect"
QT_MOC_LITERAL(17, 273, 3), // "hue"
QT_MOC_LITERAL(18, 277, 8), // "py_q_hue"
QT_MOC_LITERAL(19, 286, 12), // "isFullScreen"
QT_MOC_LITERAL(20, 299, 17), // "py_q_isFullScreen"
QT_MOC_LITERAL(21, 317, 10), // "nativeSize"
QT_MOC_LITERAL(22, 328, 15), // "py_q_nativeSize"
QT_MOC_LITERAL(23, 344, 7), // "repaint"
QT_MOC_LITERAL(24, 352, 12), // "py_q_repaint"
QT_MOC_LITERAL(25, 365, 10), // "saturation"
QT_MOC_LITERAL(26, 376, 15), // "py_q_saturation"
QT_MOC_LITERAL(27, 392, 18), // "setAspectRatioMode"
QT_MOC_LITERAL(28, 411, 4), // "mode"
QT_MOC_LITERAL(29, 416, 23), // "py_q_setAspectRatioMode"
QT_MOC_LITERAL(30, 440, 13), // "setBrightness"
QT_MOC_LITERAL(31, 454, 18), // "py_q_setBrightness"
QT_MOC_LITERAL(32, 473, 11), // "setContrast"
QT_MOC_LITERAL(33, 485, 16), // "py_q_setContrast"
QT_MOC_LITERAL(34, 502, 14), // "setDisplayRect"
QT_MOC_LITERAL(35, 517, 4), // "rect"
QT_MOC_LITERAL(36, 522, 19), // "py_q_setDisplayRect"
QT_MOC_LITERAL(37, 542, 13), // "setFullScreen"
QT_MOC_LITERAL(38, 556, 10), // "fullScreen"
QT_MOC_LITERAL(39, 567, 18), // "py_q_setFullScreen"
QT_MOC_LITERAL(40, 586, 6), // "setHue"
QT_MOC_LITERAL(41, 593, 11), // "py_q_setHue"
QT_MOC_LITERAL(42, 605, 13), // "setSaturation"
QT_MOC_LITERAL(43, 619, 18), // "py_q_setSaturation"
QT_MOC_LITERAL(44, 638, 8), // "setWinId"
QT_MOC_LITERAL(45, 647, 3), // "WId"
QT_MOC_LITERAL(46, 651, 2), // "id"
QT_MOC_LITERAL(47, 654, 13), // "py_q_setWinId"
QT_MOC_LITERAL(48, 668, 5), // "winId"
QT_MOC_LITERAL(49, 674, 10) // "py_q_winId"

    },
    "PythonQtWrapper_QVideoWindowControl\0"
    "new_QVideoWindowControl\0QVideoWindowControl*\0"
    "\0parent\0delete_QVideoWindowControl\0"
    "obj\0aspectRatioMode\0Qt::AspectRatioMode\0"
    "theWrappedObject\0py_q_aspectRatioMode\0"
    "brightness\0py_q_brightness\0contrast\0"
    "py_q_contrast\0displayRect\0py_q_displayRect\0"
    "hue\0py_q_hue\0isFullScreen\0py_q_isFullScreen\0"
    "nativeSize\0py_q_nativeSize\0repaint\0"
    "py_q_repaint\0saturation\0py_q_saturation\0"
    "setAspectRatioMode\0mode\0py_q_setAspectRatioMode\0"
    "setBrightness\0py_q_setBrightness\0"
    "setContrast\0py_q_setContrast\0"
    "setDisplayRect\0rect\0py_q_setDisplayRect\0"
    "setFullScreen\0fullScreen\0py_q_setFullScreen\0"
    "setHue\0py_q_setHue\0setSaturation\0"
    "py_q_setSaturation\0setWinId\0WId\0id\0"
    "py_q_setWinId\0winId\0py_q_winId"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_PythonQtWrapper_QVideoWindowControl[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      39,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,  209,    3, 0x0a /* Public */,
       1,    0,  212,    3, 0x2a /* Public | MethodCloned */,
       5,    1,  213,    3, 0x0a /* Public */,
       7,    1,  216,    3, 0x0a /* Public */,
      10,    1,  219,    3, 0x0a /* Public */,
      11,    1,  222,    3, 0x0a /* Public */,
      12,    1,  225,    3, 0x0a /* Public */,
      13,    1,  228,    3, 0x0a /* Public */,
      14,    1,  231,    3, 0x0a /* Public */,
      15,    1,  234,    3, 0x0a /* Public */,
      16,    1,  237,    3, 0x0a /* Public */,
      17,    1,  240,    3, 0x0a /* Public */,
      18,    1,  243,    3, 0x0a /* Public */,
      19,    1,  246,    3, 0x0a /* Public */,
      20,    1,  249,    3, 0x0a /* Public */,
      21,    1,  252,    3, 0x0a /* Public */,
      22,    1,  255,    3, 0x0a /* Public */,
      23,    1,  258,    3, 0x0a /* Public */,
      24,    1,  261,    3, 0x0a /* Public */,
      25,    1,  264,    3, 0x0a /* Public */,
      26,    1,  267,    3, 0x0a /* Public */,
      27,    2,  270,    3, 0x0a /* Public */,
      29,    2,  275,    3, 0x0a /* Public */,
      30,    2,  280,    3, 0x0a /* Public */,
      31,    2,  285,    3, 0x0a /* Public */,
      32,    2,  290,    3, 0x0a /* Public */,
      33,    2,  295,    3, 0x0a /* Public */,
      34,    2,  300,    3, 0x0a /* Public */,
      36,    2,  305,    3, 0x0a /* Public */,
      37,    2,  310,    3, 0x0a /* Public */,
      39,    2,  315,    3, 0x0a /* Public */,
      40,    2,  320,    3, 0x0a /* Public */,
      41,    2,  325,    3, 0x0a /* Public */,
      42,    2,  330,    3, 0x0a /* Public */,
      43,    2,  335,    3, 0x0a /* Public */,
      44,    2,  340,    3, 0x0a /* Public */,
      47,    2,  345,    3, 0x0a /* Public */,
      48,    1,  350,    3, 0x0a /* Public */,
      49,    1,  353,    3, 0x0a /* Public */,

 // slots: parameters
    0x80000000 | 2, QMetaType::QObjectStar,    4,
    0x80000000 | 2,
    QMetaType::Void, 0x80000000 | 2,    6,
    0x80000000 | 8, 0x80000000 | 2,    9,
    0x80000000 | 8, 0x80000000 | 2,    9,
    QMetaType::Int, 0x80000000 | 2,    9,
    QMetaType::Int, 0x80000000 | 2,    9,
    QMetaType::Int, 0x80000000 | 2,    9,
    QMetaType::Int, 0x80000000 | 2,    9,
    QMetaType::QRect, 0x80000000 | 2,    9,
    QMetaType::QRect, 0x80000000 | 2,    9,
    QMetaType::Int, 0x80000000 | 2,    9,
    QMetaType::Int, 0x80000000 | 2,    9,
    QMetaType::Bool, 0x80000000 | 2,    9,
    QMetaType::Bool, 0x80000000 | 2,    9,
    QMetaType::QSize, 0x80000000 | 2,    9,
    QMetaType::QSize, 0x80000000 | 2,    9,
    QMetaType::Void, 0x80000000 | 2,    9,
    QMetaType::Void, 0x80000000 | 2,    9,
    QMetaType::Int, 0x80000000 | 2,    9,
    QMetaType::Int, 0x80000000 | 2,    9,
    QMetaType::Void, 0x80000000 | 2, 0x80000000 | 8,    9,   28,
    QMetaType::Void, 0x80000000 | 2, 0x80000000 | 8,    9,   28,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Int,    9,   11,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Int,    9,   11,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Int,    9,   13,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Int,    9,   13,
    QMetaType::Void, 0x80000000 | 2, QMetaType::QRect,    9,   35,
    QMetaType::Void, 0x80000000 | 2, QMetaType::QRect,    9,   35,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Bool,    9,   38,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Bool,    9,   38,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Int,    9,   17,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Int,    9,   17,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Int,    9,   25,
    QMetaType::Void, 0x80000000 | 2, QMetaType::Int,    9,   25,
    QMetaType::Void, 0x80000000 | 2, 0x80000000 | 45,    9,   46,
    QMetaType::Void, 0x80000000 | 2, 0x80000000 | 45,    9,   46,
    0x80000000 | 45, 0x80000000 | 2,    9,
    0x80000000 | 45, 0x80000000 | 2,    9,

       0        // eod
};

void PythonQtWrapper_QVideoWindowControl::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        PythonQtWrapper_QVideoWindowControl *_t = static_cast<PythonQtWrapper_QVideoWindowControl *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: { QVideoWindowControl* _r = _t->new_QVideoWindowControl((*reinterpret_cast< QObject*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QVideoWindowControl**>(_a[0]) = _r; }  break;
        case 1: { QVideoWindowControl* _r = _t->new_QVideoWindowControl();
            if (_a[0]) *reinterpret_cast< QVideoWindowControl**>(_a[0]) = _r; }  break;
        case 2: _t->delete_QVideoWindowControl((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1]))); break;
        case 3: { Qt::AspectRatioMode _r = _t->aspectRatioMode((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< Qt::AspectRatioMode*>(_a[0]) = _r; }  break;
        case 4: { Qt::AspectRatioMode _r = _t->py_q_aspectRatioMode((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< Qt::AspectRatioMode*>(_a[0]) = _r; }  break;
        case 5: { int _r = _t->brightness((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 6: { int _r = _t->py_q_brightness((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 7: { int _r = _t->contrast((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 8: { int _r = _t->py_q_contrast((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 9: { QRect _r = _t->displayRect((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QRect*>(_a[0]) = _r; }  break;
        case 10: { QRect _r = _t->py_q_displayRect((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QRect*>(_a[0]) = _r; }  break;
        case 11: { int _r = _t->hue((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 12: { int _r = _t->py_q_hue((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 13: { bool _r = _t->isFullScreen((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        case 14: { bool _r = _t->py_q_isFullScreen((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        case 15: { QSize _r = _t->nativeSize((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QSize*>(_a[0]) = _r; }  break;
        case 16: { QSize _r = _t->py_q_nativeSize((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QSize*>(_a[0]) = _r; }  break;
        case 17: _t->repaint((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1]))); break;
        case 18: _t->py_q_repaint((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1]))); break;
        case 19: { int _r = _t->saturation((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 20: { int _r = _t->py_q_saturation((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 21: _t->setAspectRatioMode((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])),(*reinterpret_cast< Qt::AspectRatioMode(*)>(_a[2]))); break;
        case 22: _t->py_q_setAspectRatioMode((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])),(*reinterpret_cast< Qt::AspectRatioMode(*)>(_a[2]))); break;
        case 23: _t->setBrightness((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 24: _t->py_q_setBrightness((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 25: _t->setContrast((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 26: _t->py_q_setContrast((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 27: _t->setDisplayRect((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])),(*reinterpret_cast< const QRect(*)>(_a[2]))); break;
        case 28: _t->py_q_setDisplayRect((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])),(*reinterpret_cast< const QRect(*)>(_a[2]))); break;
        case 29: _t->setFullScreen((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])),(*reinterpret_cast< bool(*)>(_a[2]))); break;
        case 30: _t->py_q_setFullScreen((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])),(*reinterpret_cast< bool(*)>(_a[2]))); break;
        case 31: _t->setHue((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 32: _t->py_q_setHue((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 33: _t->setSaturation((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 34: _t->py_q_setSaturation((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 35: _t->setWinId((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])),(*reinterpret_cast< WId(*)>(_a[2]))); break;
        case 36: _t->py_q_setWinId((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])),(*reinterpret_cast< WId(*)>(_a[2]))); break;
        case 37: { WId _r = _t->winId((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< WId*>(_a[0]) = _r; }  break;
        case 38: { WId _r = _t->py_q_winId((*reinterpret_cast< QVideoWindowControl*(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< WId*>(_a[0]) = _r; }  break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 2:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 3:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 4:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 5:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 6:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 7:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 8:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 9:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 10:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 11:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 12:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 13:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 14:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 15:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 16:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 17:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 18:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 19:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 20:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 21:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 22:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 23:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 24:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 25:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 26:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 27:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 28:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 29:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 30:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 31:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 32:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 33:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 34:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 35:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 36:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 37:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        case 38:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVideoWindowControl* >(); break;
            }
            break;
        }
    }
}

const QMetaObject PythonQtWrapper_QVideoWindowControl::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_PythonQtWrapper_QVideoWindowControl.data,
      qt_meta_data_PythonQtWrapper_QVideoWindowControl,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *PythonQtWrapper_QVideoWindowControl::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *PythonQtWrapper_QVideoWindowControl::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_PythonQtWrapper_QVideoWindowControl.stringdata0))
        return static_cast<void*>(const_cast< PythonQtWrapper_QVideoWindowControl*>(this));
    return QObject::qt_metacast(_clname);
}

int PythonQtWrapper_QVideoWindowControl::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 39)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 39;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 39)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 39;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
