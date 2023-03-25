#pragma once
#include <stdexcept>
#include <sstream>
#include <string>

template <typename Arg>
inline void format_args(std::ostream& os, Arg const& arg)
{
  os << arg;
}

template <typename Arg, typename... Args>
inline void format_args(std::ostream& os, Arg const& arg, Args const&... args)
{
  os << arg;
  format_args(os, args...);
}

template <typename... Args>
[[nodiscard]] inline std::string format_str(Args const&... args)
{
  std::stringstream ss;
  format_args(ss, args...);
  return ss.str();
}

template <class Exception, typename... Args>
[[noreturn]] inline void throw_typed(Args const&... args)
{
  std::stringstream ss;
  format_args(ss, args...);
  throw Exception(ss.str());
}

template <typename... Args>
[[noreturn]] inline void throw_runtime_error(Args const&... args)
{
  throw_typed<std::runtime_error>(args...);
}

template <typename... Args>
[[noreturn]] inline void throw_invalid_argument(Args const&... args)
{
  throw_typed<std::invalid_argument>(args...);
}

template <typename... Args>
[[noreturn]] inline void throw_length_error(Args const&... args)
{
  throw_typed<std::length_error>(args...);
}
