#ifndef UNITREE_EXAMPLE_BEHAVIOR_DIR_HPP_
#define UNITREE_EXAMPLE_BEHAVIOR_DIR_HPP_

#include <limits.h>
#include <unistd.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace unitree_example {

// 当前可执行文件所在目录；Linux 上通过 /proc/self/exe 获取，
// 不依赖启动时的工作目录，也不把构建期的源码路径写进程序。
inline std::string ExecutableDirectory() {
#if defined(__linux__)
    char path[PATH_MAX] = {};
    const ssize_t len = ::readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len <= 0) {
        return std::string();
    }
    path[len] = '\0';
    const std::string exe(path);
    const std::string::size_type slash = exe.find_last_of('/');
    if (slash == std::string::npos) {
        return std::string(".");
    }
    return exe.substr(0, slash);
#else
    // 非 Linux 平台没有 /proc/self/exe，需显式设置 UNITREE_SDK2_BEHAVIOR_DIR。
    return std::string();
#endif
}

inline bool DirectoryExists(const std::string& path) {
    return ::access(path.c_str(), R_OK | X_OK) == 0;
}

// 解析 <behavior_lib>/<robot>/ 的运行时位置，优先级：
//   1. $UNITREE_SDK2_BEHAVIOR_DIR/<robot>
//      （由 <prefix>/unitree_sdk2-env.sh 导出，也可手动指向别处的数据副本）
//   2. <可执行文件目录>/../share/unitree_sdk2/behavior_lib/<robot>
//      （安装前缀里的默认布局，配合 $ORIGIN 相对定位，前缀可整体搬运）
inline std::string ResolveBehaviorDir(const std::string& robot) {
    const char* env = std::getenv("UNITREE_SDK2_BEHAVIOR_DIR");
    if (env != nullptr && env[0] != '\0') {
        std::string candidate(env);
        if (candidate.back() != '/') {
            candidate += '/';
        }
        candidate += robot + '/';
        if (DirectoryExists(candidate)) {
            return candidate;
        }
    }

    const std::string exe_dir = ExecutableDirectory();
    if (!exe_dir.empty()) {
        const std::string candidate =
            exe_dir + "/../share/unitree_sdk2/behavior_lib/" + robot + '/';
        if (DirectoryExists(candidate)) {
            return candidate;
        }
    }

    throw std::runtime_error(
        "cannot locate behavior_lib/" + robot +
        "; source <prefix>/unitree_sdk2-env.sh or set UNITREE_SDK2_BEHAVIOR_DIR");
}

}  // namespace unitree_example

#endif  // UNITREE_EXAMPLE_BEHAVIOR_DIR_HPP_
