#pragma once

#include "../observer/tfprof.hpp"
#include "taskflow.hpp"
#include "async_task.hpp"

/**
@file executor.hpp
@brief executor include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// Executor Definition
// ----------------------------------------------------------------------------

/** 
@class Executor

@brief class to create an executor 

An tf::Executor manages a set of worker threads to run tasks 
using an efficient work-stealing scheduling algorithm.

@code{.cpp}
// Declare an executor and a taskflow
tf::Executor executor;
tf::Taskflow taskflow;

// Add three tasks into the taskflow
tf::Task A = taskflow.emplace([] () { std::cout << "This is TaskA\n"; });
tf::Task B = taskflow.emplace([] () { std::cout << "This is TaskB\n"; });
tf::Task C = taskflow.emplace([] () { std::cout << "This is TaskC\n"; });

// Build precedence between tasks
A.precede(B, C);

tf::Future<void> fu = executor.run(taskflow);
fu.wait();  // block until the execution completes

executor.run(taskflow, [](){ std::cout << "end of 1 run"; }).wait();
executor.run_n(taskflow, 4);
executor.wait_for_all();  // block until all associated executions finish
executor.run_n(taskflow, 4, [](){ std::cout << "end of 4 runs"; }).wait();
executor.run_until(taskflow, [cnt=0] () mutable { return ++cnt == 10; });
@endcode

Most executor methods are @em thread-safe. 
For example, you can submit multiple taskflows to an executor concurrently 
from different threads, while other threads simultaneously create asynchronous tasks.

@code{.cpp}
std::thread t1([&](){ executor.run(taskflow); };
std::thread t2([&](){ executor.async([](){ std::cout << "async task from t2\n"; }); });
executor.async([&](){ std::cout << "async task from the main thread\n"; });
@endcode

@note
To know more about tf::Executor, please refer to @ref ExecuteTaskflow.
*/
class Executor {

  friend class FlowBuilder;
  friend class Subflow;
  friend class Runtime;
  friend class NonpreemptiveRuntime;
  friend class Algorithm;
  friend class TaskGroup;

  public:

  /**
  @brief constructs the executor with @c N worker threads

  @param N number of workers (default std::thread::hardware_concurrency)
  @param wif interface class instance to configure workers' behaviors

  The constructor spawns @c N worker threads to run tasks in a
  work-stealing loop. The number of workers must be greater than zero
  or an exception will be thrown.
  By default, the number of worker threads is equal to the maximum
  hardware concurrency returned by std::thread::hardware_concurrency.

  Users can alter the worker behavior, such as changing thread affinity,
  via deriving an instance from tf::WorkerInterface.

  @note
  An exception will be thrown if executor construction fails.
  */
  explicit Executor(
    size_t N = std::thread::hardware_concurrency(),
    std::shared_ptr<WorkerInterface> wif = nullptr
  );

  /**
  @brief destructs the executor

  The destructor calls Executor::wait_for_all to wait for all submitted
  taskflows to complete and then notifies all worker threads to stop
  and join these threads.
  */
  ~Executor();

  /**
  @brief runs a taskflow once

  @param taskflow a tf::Taskflow object

  @return a tf::Future that holds the result of the execution

  This member function executes the given taskflow once and returns a tf::Future
  object that eventually holds the result of the execution.

  @code{.cpp}
  tf::Future<void> future = executor.run(taskflow);
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  tf::Future<void> run(Taskflow& taskflow);

  /**
  @brief runs a moved taskflow once

  @param taskflow a moved tf::Taskflow object

  @return a tf::Future that holds the result of the execution

  This member function executes a moved taskflow once and returns a tf::Future
  object that eventually holds the result of the execution.
  The executor will take care of the lifetime of the moved taskflow.

  @code{.cpp}
  tf::Future<void> future = executor.run(std::move(taskflow));
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  tf::Future<void> run(Taskflow&& taskflow);

  /**
  @brief runs a taskflow once and invoke a callback upon completion

  @param taskflow a tf::Taskflow object
  @param callable a callable object to be invoked after this run

  @return a tf::Future that holds the result of the execution

  This member function executes the given taskflow once and invokes the given
  callable when the execution completes.
  This member function returns a tf::Future object that
  eventually holds the result of the execution.

  @code{.cpp}
  tf::Future<void> future = executor.run(taskflow, [](){ std::cout << "done"; });
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.

  */
  template<typename C>
  tf::Future<void> run(Taskflow& taskflow, C&& callable);

  /**
  @brief runs a moved taskflow once and invoke a callback upon completion

  @param taskflow a moved tf::Taskflow object
  @param callable a callable object to be invoked after this run

  @return a tf::Future that holds the result of the execution

  This member function executes a moved taskflow once and invokes the given
  callable when the execution completes.
  This member function returns a tf::Future object that
  eventually holds the result of the execution.
  The executor will take care of the lifetime of the moved taskflow.

  @code{.cpp}
  tf::Future<void> future = executor.run(
    std::move(taskflow), [](){ std::cout << "done"; }
  );
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  template<typename C>
  tf::Future<void> run(Taskflow&& taskflow, C&& callable);

  /**
  @brief runs a taskflow for @c N times

  @param taskflow a tf::Taskflow object
  @param N number of runs

  @return a tf::Future that holds the result of the execution

  This member function executes the given taskflow @c N times and returns a tf::Future
  object that eventually holds the result of the execution.

  @code{.cpp}
  tf::Future<void> future = executor.run_n(taskflow, 2);  // run taskflow 2 times
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  tf::Future<void> run_n(Taskflow& taskflow, size_t N);

  /**
  @brief runs a moved taskflow for @c N times

  @param taskflow a moved tf::Taskflow object
  @param N number of runs

  @return a tf::Future that holds the result of the execution

  This member function executes a moved taskflow @c N times and returns a tf::Future
  object that eventually holds the result of the execution.
  The executor will take care of the lifetime of the moved taskflow.

  @code{.cpp}
  tf::Future<void> future = executor.run_n(
    std::move(taskflow), 2    // run the moved taskflow 2 times
  );
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  tf::Future<void> run_n(Taskflow&& taskflow, size_t N);

  /**
  @brief runs a taskflow for @c N times and then invokes a callback

  @param taskflow a tf::Taskflow
  @param N number of runs
  @param callable a callable object to be invoked after this run

  @return a tf::Future that holds the result of the execution

  This member function executes the given taskflow @c N times and invokes the given
  callable when the execution completes.
  This member function returns a tf::Future object that
  eventually holds the result of the execution.

  @code{.cpp}
  tf::Future<void> future = executor.run(
    taskflow, 2, [](){ std::cout << "done"; }  // runs taskflow 2 times and invoke
                                               // the lambda to print "done"
  );
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  template<typename C>
  tf::Future<void> run_n(Taskflow& taskflow, size_t N, C&& callable);

  /**
  @brief runs a moved taskflow for @c N times and then invokes a callback

  @param taskflow a moved tf::Taskflow
  @param N number of runs
  @param callable a callable object to be invoked after this run

  @return a tf::Future that holds the result of the execution

  This member function executes a moved taskflow @c N times and invokes the given
  callable when the execution completes.
  This member function returns a tf::Future object that
  eventually holds the result of the execution.

  @code{.cpp}
  tf::Future<void> future = executor.run_n(
    // run the moved taskflow 2 times and invoke the lambda to print "done"
    std::move(taskflow), 2, [](){ std::cout << "done"; }
  );
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  template<typename C>
  tf::Future<void> run_n(Taskflow&& taskflow, size_t N, C&& callable);

  /**
  @brief runs a taskflow multiple times until the predicate becomes true

  @param taskflow a tf::Taskflow
  @param pred a boolean predicate to return @c true for stop

  @return a tf::Future that holds the result of the execution

  This member function executes the given taskflow multiple times until
  the predicate returns @c true.
  This member function returns a tf::Future object that
  eventually holds the result of the execution.

  @code{.cpp}
  tf::Future<void> future = executor.run_until(
    taskflow, [](){ return rand()%10 == 0 }
  );
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  template<typename P>
  tf::Future<void> run_until(Taskflow& taskflow, P&& pred);

  /**
  @brief runs a moved taskflow and keeps running it
         until the predicate becomes true

  @param taskflow a moved tf::Taskflow object
  @param pred a boolean predicate to return @c true for stop

  @return a tf::Future that holds the result of the execution

  This member function executes a moved taskflow multiple times until
  the predicate returns @c true.
  This member function returns a tf::Future object that
  eventually holds the result of the execution.
  The executor will take care of the lifetime of the moved taskflow.

  @code{.cpp}
  tf::Future<void> future = executor.run_until(
    std::move(taskflow), [](){ return rand()%10 == 0 }
  );
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  template<typename P>
  tf::Future<void> run_until(Taskflow&& taskflow, P&& pred);

  /**
  @brief runs a taskflow multiple times until the predicate becomes true and
         then invokes the callback

  @param taskflow a tf::Taskflow
  @param pred a boolean predicate to return @c true for stop
  @param callable a callable object to be invoked after this run completes

  @return a tf::Future that holds the result of the execution

  This member function executes the given taskflow multiple times until
  the predicate returns @c true and then invokes the given callable when
  the execution completes.
  This member function returns a tf::Future object that
  eventually holds the result of the execution.

  @code{.cpp}
  tf::Future<void> future = executor.run_until(
    taskflow, [](){ return rand()%10 == 0 }, [](){ std::cout << "done"; }
  );
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  template<typename P, typename C>
  tf::Future<void> run_until(Taskflow& taskflow, P&& pred, C&& callable);

  /**
  @brief runs a moved taskflow and keeps running
         it until the predicate becomes true and then invokes the callback

  @param taskflow a moved tf::Taskflow
  @param pred a boolean predicate to return @c true for stop
  @param callable a callable object to be invoked after this run completes

  @return a tf::Future that holds the result of the execution

  This member function executes a moved taskflow multiple times until
  the predicate returns @c true and then invokes the given callable when
  the execution completes.
  This member function returns a tf::Future object that
  eventually holds the result of the execution.
  The executor will take care of the lifetime of the moved taskflow.

  @code{.cpp}
  tf::Future<void> future = executor.run_until(
    std::move(taskflow),
    [](){ return rand()%10 == 0 }, [](){ std::cout << "done"; }
  );
  // do something else
  future.wait();
  @endcode

  This member function is thread-safe.
  */
  template<typename P, typename C>
  tf::Future<void> run_until(Taskflow&& taskflow, P&& pred, C&& callable);

  /**
  @brief runs a target graph and waits until it completes using 
         an internal worker of this executor
  
  @tparam T target type which is either convertible to `tf::Graph` or has `tf::Graph& T::graph()` defined
  @param target the target task graph object

  The method coruns a target graph cooperatively with other workers in the same executor
  and block until the execution completes.
  Under cooperative execution, a worker is not preempted. Instead, it continues 
  participating in the work-stealing loop, executing available tasks alongside 
  other workers.  
  
  @code{.cpp}
  tf::Executor executor(2);
  tf::Taskflow taskflow;
  std::array<tf::Taskflow, 1000> others;
  
  std::atomic<size_t> counter{0};
  
  for(size_t n=0; n<1000; n++) {
    for(size_t i=0; i<1000; i++) {
      others[n].emplace([&](){ counter++; });
    }
    taskflow.emplace([&executor, &tf=others[n]](){
      executor.corun(tf);
      //executor.run(tf).wait();  <- blocking the worker without doing anything
      //                             will introduce deadlock
    });
  }
  executor.run(taskflow).wait();
  @endcode 

  The method is thread-safe as long as the target is not concurrently
  ran by two or more threads.

  @note
  You must call tf::Executor::corun from a worker of the calling executor
  or an exception will be thrown.
  */
  template <typename T>
  void corun(T& target);

  /**
  @brief keeps running the work-stealing loop until the predicate returns `true`
  
  @tparam P predicate type
  @param predicate a boolean predicate to indicate when to stop the loop

  The method keeps the caller worker running in the work-stealing loop
  until the stop predicate becomes true.

  The method keeps the calling worker running available tasks cooperatively 
  with other workers in the same executor and block until the predicate return `true`.
  Under cooperative execution, a worker is not preempted. Instead, it continues 
  participating in the work-stealing loop, executing available tasks alongside 
  other workers.  

  @code{.cpp}
  taskflow.emplace([&](){
    std::future<void> fu = std::async([](){ std::sleep(100s); });
    executor.corun_until([](){
      return fu.wait_for(std::chrono::seconds(0)) == future_status::ready;
    });
  });
  @endcode

  @note
  You must call tf::Executor::corun_until from a worker of the calling executor
  or an exception will be thrown.
  */
  template <typename P>
  void corun_until(P&& predicate);

  /**
  @brief waits for all tasks to complete

  This member function waits until all submitted tasks
  (e.g., taskflows, asynchronous tasks) to finish.

  @code{.cpp}
  executor.run(taskflow1);
  executor.run_n(taskflow2, 10);
  executor.run_n(taskflow3, 100);
  executor.wait_for_all();  // wait until the above submitted taskflows finish
  @endcode
  */
  void wait_for_all();

  /**
  @brief queries the number of worker threads

  Each worker represents a unique thread spawned by an executor upon its construction time.

  @code{.cpp}
  tf::Executor executor(4);
  std::cout << executor.num_workers();    // 4
  @endcode
  */
  size_t num_workers() const noexcept;
  
  /**
  @brief queries the number of workers that are in the waiting loop

  A worker in the waiting loop has exhausted its local queue and made enough stealing attempts,
  and is now ready to be preempted and enter the waiting state.
  */
  size_t num_waiters() const noexcept;
  
  /**
  @brief queries the number of work-stealing queues used by the executor
  */
  size_t num_queues() const noexcept;

  /**
  @brief queries the number of running topologies at the time of this call

  When a taskflow is submitted to an executor, a topology is created to store
  runtime metadata of the running taskflow.
  When the execution of the submitted taskflow finishes,
  its corresponding topology will be removed from the executor.

  @code{.cpp}
  executor.run(taskflow);
  std::cout << executor.num_topologies();  // 0 or 1 (taskflow still running)
  @endcode
  */
  size_t num_topologies() const;

  /**
  @brief queries pointer to the calling worker if it belongs to this executor, otherwise returns `nullptr`

  Returns a pointer to the per-worker storage associated with this executor. 
  If the calling thread is not a worker of this executor, the function returns `nullptr`.

  @code{.cpp}
  auto w = executor.this_worker();
  tf::Taskflow taskflow;
  tf::Executor executor;
  executor.async([&](){
    assert(executor.this_worker() != nullptr);
    assert(executor.this_worker()->executor() == &executor);
  });
  @endcode
  */
  Worker* this_worker();
  
  /**
  @brief queries the id of the caller thread within this executor

  Each worker has an unique id in the range of @c 0 to @c N-1 associated with
  its parent executor.
  If the caller thread does not belong to the executor, @c -1 is returned.

  @code{.cpp}
  tf::Executor executor(4);   // 4 workers in the executor
  executor.this_worker_id();  // -1 (main thread is not a worker)

  taskflow.emplace([&](){
    std::cout << executor.this_worker_id();  // 0, 1, 2, or 3
  });
  executor.run(taskflow);
  @endcode
  */
  int this_worker_id() const;
 
  // --------------------------------------------------------------------------
  // Observer methods
  // --------------------------------------------------------------------------

  /**
  @brief constructs an observer to inspect the activities of worker threads

  @tparam Observer observer type derived from tf::ObserverInterface
  @tparam ArgsT argument parameter pack

  @param args arguments to forward to the constructor of the observer

  @return a shared pointer to the created observer

  Each executor manages a list of observers with shared ownership with callers.
  For each of these observers, the two member functions,
  tf::ObserverInterface::on_entry and tf::ObserverInterface::on_exit
  will be called before and after the execution of a task.

  This member function is not thread-safe.
  */
  template <typename Observer, typename... ArgsT>
  std::shared_ptr<Observer> make_observer(ArgsT&&... args);

  /**
  @brief removes an observer from the executor

  This member function is not thread-safe.
  */
  template <typename Observer>
  void remove_observer(std::shared_ptr<Observer> observer);

  /**
  @brief queries the number of observers
  */
  size_t num_observers() const noexcept;

  // --------------------------------------------------------------------------
  // Async Task Methods
  // --------------------------------------------------------------------------
  
  /**
  @brief creates a parameterized asynchronous task to run the given function

  @tparam P task parameter type satisfying tf::TaskParamsLike
  @tparam F callable type

  @param params task parameters
  @param func callable object

  @return a @std_future that will hold the result of the execution
  
  The method creates a parameterized asynchronous task 
  to run the given function and return a @std_future object 
  that eventually will hold the result of the execution.

  @code{.cpp}
  std::future<int> future = executor.async("name", [](){
    std::cout << "create an asynchronous task with a name and returns 1\n";
    return 1;
  });
  future.get();
  @endcode

  This member function is thread-safe.
  */
  template <TaskParamsLike P, typename F>
  auto async(P&& params, F&& func);

  /**
  @brief runs a given function asynchronously

  @tparam F callable type

  @param func callable object

  @return a @std_future that will hold the result of the execution

  The method creates an asynchronous task to run the given function
  and return a @std_future object that eventually will hold the result
  of the return value.

  @code{.cpp}
  std::future<int> future = executor.async([](){
    std::cout << "create an asynchronous task and returns 1\n";
    return 1;
  });
  future.get();
  @endcode

  This member function is thread-safe.
  */
  template <typename F>
  auto async(F&& func);

  /**
  @brief similar to tf::Executor::async but does not return a future object

  @tparam P task parameter type satisfying tf::TaskParamsLike
  @tparam F callable type

  @param params task parameters
  @param func callable object

  The method creates a parameterized asynchronous task 
  to run the given function without returning any @std_future object.
  This member function is more efficient than tf::Executor::async 
  and is encouraged to use when applications do not need a @std_future to acquire
  the result or synchronize the execution.

  @code{.cpp}
  executor.silent_async("name", [](){
    std::cout << "create an asynchronous task with a name and no return\n";
  });
  executor.wait_for_all();
  @endcode

  This member function is thread-safe.
  */
  template <TaskParamsLike P, typename F>
  void silent_async(P&& params, F&& func);
  
  /**
  @brief similar to tf::Executor::async but does not return a future object
  
  @tparam F callable type
  
  @param func callable object

  The method creates an asynchronous task 
  to run the given function without returning any @std_future object.
  This member function is more efficient than tf::Executor::async 
  and is encouraged to use when applications do not need a @std_future to acquire
  the result or synchronize the execution.

  @code{.cpp}
  executor.silent_async([](){
    std::cout << "create an asynchronous task with no return\n";
  });
  executor.wait_for_all();
  @endcode

  This member function is thread-safe.
  */
  template <typename F>
  void silent_async(F&& func);

  // --------------------------------------------------------------------------
  // Silent Dependent Async Methods
  // --------------------------------------------------------------------------
  
  /**
  @brief runs the given function asynchronously 
         when the given predecessors finish

  @tparam F callable type
  @tparam Tasks task types convertible to tf::AsyncTask

  @param func callable object
  @param tasks asynchronous tasks on which this execution depends
  
  @return a tf::AsyncTask handle 
  
  This member function is more efficient than tf::Executor::dependent_async
  and is encouraged to use when you do not want a @std_future to
  acquire the result or synchronize the execution.
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.

  @code{.cpp}
  tf::AsyncTask A = executor.silent_dependent_async([](){ printf("A\n"); });
  tf::AsyncTask B = executor.silent_dependent_async([](){ printf("B\n"); });
  executor.silent_dependent_async([](){ printf("C runs after A and B\n"); }, A, B);
  executor.wait_for_all();
  @endcode

  This member function is thread-safe.
  */
  template <typename F, AsyncTaskHandleLike... Tasks>
  tf::AsyncTask silent_dependent_async(F&& func, Tasks&&... tasks);
  
  /**
  @brief runs the given function asynchronously 
         when the given predecessors finish
  
  @tparam P task parameter type satisfying tf::TaskParamsLike
  @tparam F callable type
  @tparam Tasks task types convertible to tf::AsyncTask

  @param params task parameters
  @param func callable object
  @param tasks asynchronous tasks on which this execution depends
  
  @return a tf::AsyncTask handle 
  
  This member function is more efficient than tf::Executor::dependent_async
  and is encouraged to use when you do not want a @std_future to
  acquire the result or synchronize the execution.
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  Assigned task names will appear in the observers of the executor.

  @code{.cpp}
  tf::AsyncTask A = executor.silent_dependent_async("A", [](){ printf("A\n"); });
  tf::AsyncTask B = executor.silent_dependent_async("B", [](){ printf("B\n"); });
  executor.silent_dependent_async(
    "C", [](){ printf("C runs after A and B\n"); }, A, B
  );
  executor.wait_for_all();
  @endcode

  This member function is thread-safe.
  */
  template <TaskParamsLike P, typename F, AsyncTaskHandleLike... Tasks>
  tf::AsyncTask silent_dependent_async(P&& params, F&& func, Tasks&&... tasks);
  
  /**
  @brief runs the given function asynchronously 
         when the given range of predecessors finish
  
  @tparam F callable type
  @tparam I iterator type 

  @param func callable object
  @param first iterator to the beginning (inclusive)
  @param last iterator to the end (exclusive)
  
  @return a tf::AsyncTask handle 
  
  This member function is more efficient than tf::Executor::dependent_async
  and is encouraged to use when you do not want a @std_future to
  acquire the result or synchronize the execution.
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.

  @code{.cpp}
  std::array<tf::AsyncTask, 2> array {
    executor.silent_dependent_async([](){ printf("A\n"); }),
    executor.silent_dependent_async([](){ printf("B\n"); })
  };
  executor.silent_dependent_async(
    [](){ printf("C runs after A and B\n"); }, array.begin(), array.end()
  );
  executor.wait_for_all();
  @endcode

  This member function is thread-safe.
  */
  template <typename F, std::input_iterator I>
  tf::AsyncTask silent_dependent_async(F&& func, I first, I last);
  
  /**
  @brief runs the given function asynchronously 
         when the given range of predecessors finish
  
  @tparam P task parameter type satisfying tf::TaskParamsLike
  @tparam F callable type
  @tparam I iterator type 

  @param params tasks parameters
  @param func callable object
  @param first iterator to the beginning (inclusive)
  @param last iterator to the end (exclusive)

  @return a tf::AsyncTask handle 
  
  This member function is more efficient than tf::Executor::dependent_async
  and is encouraged to use when you do not want a @std_future to
  acquire the result or synchronize the execution.
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  Assigned task names will appear in the observers of the executor.

  @code{.cpp}
  std::array<tf::AsyncTask, 2> array {
    executor.silent_dependent_async("A", [](){ printf("A\n"); }),
    executor.silent_dependent_async("B", [](){ printf("B\n"); })
  };
  executor.silent_dependent_async(
    "C", [](){ printf("C runs after A and B\n"); }, array.begin(), array.end()
  );
  executor.wait_for_all();
  @endcode

  This member function is thread-safe.
  */
  template <TaskParamsLike P, typename F, std::input_iterator I>
  tf::AsyncTask silent_dependent_async(P&& params, F&& func, I first, I last);
  
  // --------------------------------------------------------------------------
  // Dependent Async Methods
  // --------------------------------------------------------------------------
  
  /**
  @brief runs the given function asynchronously 
         when the given predecessors finish
  
  @tparam F callable type
  @tparam Tasks task types convertible to tf::AsyncTask

  @param func callable object
  @param tasks asynchronous tasks on which this execution depends
  
  @return a pair of a tf::AsyncTask handle and 
                    a @std_future that holds the result of the execution
  
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  Task @c C returns a pair of its tf::AsyncTask handle and a std::future<int>
  that eventually will hold the result of the execution.

  @code{.cpp}
  tf::AsyncTask A = executor.silent_dependent_async([](){ printf("A\n"); });
  tf::AsyncTask B = executor.silent_dependent_async([](){ printf("B\n"); });
  auto [C, fuC] = executor.dependent_async(
    [](){ 
      printf("C runs after A and B\n"); 
      return 1;
    }, 
    A, B
  );
  fuC.get();  // C finishes, which in turns means both A and B finish
  @endcode

  You can mix the use of tf::AsyncTask handles 
  returned by tf::Executor::dependent_async and tf::Executor::silent_dependent_async
  when specifying task dependencies.

  This member function is thread-safe.
  */
  template <typename F, AsyncTaskHandleLike... Tasks>
  auto dependent_async(F&& func, Tasks&&... tasks);
  
  /**
  @brief runs the given function asynchronously
         when the given predecessors finish
  
  @tparam P task parameters type satisfying tf::TaskParamsLike
  @tparam F callable type
  @tparam Tasks task types convertible to tf::AsyncTask
  
  @param params task parameters
  @param func callable object
  @param tasks asynchronous tasks on which this execution depends
  
  @return a pair of a tf::AsyncTask handle and 
                    a @std_future that holds the result of the execution
  
  The example below creates three named asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  Task @c C returns a pair of its tf::AsyncTask handle and a std::future<int>
  that eventually will hold the result of the execution.
  Assigned task names will appear in the observers of the executor.

  @code{.cpp}
  tf::AsyncTask A = executor.silent_dependent_async("A", [](){ printf("A\n"); });
  tf::AsyncTask B = executor.silent_dependent_async("B", [](){ printf("B\n"); });
  auto [C, fuC] = executor.dependent_async(
    "C",
    [](){ 
      printf("C runs after A and B\n"); 
      return 1;
    }, 
    A, B
  );
  assert(fuC.get()==1);  // C finishes, which in turns means both A and B finish
  @endcode

  You can mix the use of tf::AsyncTask handles 
  returned by tf::Executor::dependent_async and tf::Executor::silent_dependent_async
  when specifying task dependencies.

  This member function is thread-safe.
  */
  template <TaskParamsLike P, typename F, AsyncTaskHandleLike... Tasks>
  auto dependent_async(P&& params, F&& func, Tasks&&... tasks);
  
  /**
  @brief runs the given function asynchronously 
         when the given range of predecessors finish
  
  @tparam F callable type
  @tparam I iterator type 

  @param func callable object
  @param first iterator to the beginning (inclusive)
  @param last iterator to the end (exclusive)
  
  @return a pair of a tf::AsyncTask handle and 
                    a @std_future that holds the result of the execution
  
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  Task @c C returns a pair of its tf::AsyncTask handle and a std::future<int>
  that eventually will hold the result of the execution.

  @code{.cpp}
  std::array<tf::AsyncTask, 2> array {
    executor.silent_dependent_async([](){ printf("A\n"); }),
    executor.silent_dependent_async([](){ printf("B\n"); })
  };
  auto [C, fuC] = executor.dependent_async(
    [](){ 
      printf("C runs after A and B\n"); 
      return 1;
    }, 
    array.begin(), array.end()
  );
  assert(fuC.get()==1);  // C finishes, which in turns means both A and B finish
  @endcode

  You can mix the use of tf::AsyncTask handles 
  returned by tf::Executor::dependent_async and tf::Executor::silent_dependent_async
  when specifying task dependencies.

  This member function is thread-safe.
  */
  template <typename F, std::input_iterator I>
  auto dependent_async(F&& func, I first, I last);
  
  /**
  @brief runs the given function asynchronously 
         when the given range of predecessors finish
  
  @tparam P task parameters type satisfying tf::TaskParamsLike
  @tparam F callable type
  @tparam I iterator type 
  
  @param params task parameters
  @param func callable object
  @param first iterator to the beginning (inclusive)
  @param last iterator to the end (exclusive)
  
  @return a pair of a tf::AsyncTask handle and 
                    a @std_future that holds the result of the execution
  
  The example below creates three named asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  Task @c C returns a pair of its tf::AsyncTask handle and a std::future<int>
  that eventually will hold the result of the execution.
  Assigned task names will appear in the observers of the executor.

  @code{.cpp}
  std::array<tf::AsyncTask, 2> array {
    executor.silent_dependent_async("A", [](){ printf("A\n"); }),
    executor.silent_dependent_async("B", [](){ printf("B\n"); })
  };
  auto [C, fuC] = executor.dependent_async(
    "C",
    [](){ 
      printf("C runs after A and B\n"); 
      return 1;
    }, 
    array.begin(), array.end()
  );
  assert(fuC.get()==1);  // C finishes, which in turns means both A and B finish
  @endcode

  You can mix the use of tf::AsyncTask handles 
  returned by tf::Executor::dependent_async and tf::Executor::silent_dependent_async
  when specifying task dependencies.

  This member function is thread-safe.
  */
  template <TaskParamsLike P, typename F, std::input_iterator I>
  auto dependent_async(P&& params, F&& func, I first, I last);

  // ----------------------------------------------------------------------------------------------
  // Task Group
  // ----------------------------------------------------------------------------------------------
  
  /**
  @brief creates a task group that executes a collection of asynchronous tasks
  @return a tf::TaskGroup object associated with the current executor
 
  A TaskGroup allows submitting multiple asynchronous tasks to the executor
  and waiting for their completion collectively using `corun()`. Tasks added
  to the group can execute in parallel and may capture local variables by value
  or reference, depending on your needs. 
  This can be useful for divide-and-conquer
  algorithms, parallel loops, or any workflow that requires grouping related tasks.
  
  Example (computing Fibonacci numbers in parallel):

  @code{.cpp}
  tf::Executor executor;
  
  size_t fibonacci(size_t N) {

    if (N < 2) return N;
  
    size_t res1, res2;
  
    // Create a task group from the current executor
    tf::TaskGroup tg = get_executor().task_group();
  
    // Submit asynchronous tasks to the group
    tg.silent_async([N, &res1](){ res1 = fibonacci(N-1); });
    res2 = fibonacci(N-2);  // compute one branch synchronously
  
    // Wait for all tasks in the group to complete
    tg.corun();
  
    return res1 + res2;
  }
  
  int main() {
    return executor.async([](){ return fibonacci(30); }).get();
  }
  @endcode
  
  This member function is thread-safe.

  @note
  Due to cooperative execution, a task group can only be created by a worker of an executor.
  */
  TaskGroup task_group();

  private:
  
  struct Buffer {
    std::mutex mutex;
    UnboundedWSQ<Node*> queue;
  };  
  
  std::vector<Worker> _workers;
  std::vector<Buffer> _buffers;
  
  // notifier's state variable and num_topologies should sit on different cachelines
  // or the false sharing can cause serious performance drop
  alignas(TF_CACHELINE_SIZE) DefaultNotifier _notifier;
  alignas(TF_CACHELINE_SIZE) std::atomic<size_t> _num_topologies {0};
  
  std::unordered_map<std::thread::id, Worker*> _t2w;
  std::unordered_set<std::shared_ptr<ObserverInterface>> _observers;

  // true when constructed with 0 workers (inline mode).
  bool _inlined {false};

  // synchronous topological run of a static graph.
  void _run_inline(Graph&);

  void _shutdown();
  void _observer_prologue(Worker&, Node*);
  void _observer_epilogue(Worker&, Node*);
  void _spawn(size_t, std::shared_ptr<WorkerInterface>);
  void _exploit_task(Worker&, Node*&);
  bool _explore_task(Worker&, Node*&);
  void _schedule(Worker&, Node*);
  void _schedule(Node*);
  void _schedule_graph(Worker&, Graph&, Topology*, NodeBase*);
  void _spill(Node*);
  void _set_up_topology(Worker*, Topology*);
  void _tear_down_topology(Worker&, Topology*, Node*&);
  void _tear_down_async(Worker&, Node*, Node*&);
  void _tear_down_dependent_async(Worker&, Node*, Node*&);
  void _tear_down_nonasync(Worker&, Node*, Node*&);
  void _tear_down_invoke(Worker&, Node*, Node*&);
  void _increment_topology();
  void _decrement_topology();
  void _invoke(Worker&, Node*);
  void _invoke_static_task(Worker&, Node*);
  void _invoke_nonpreemptive_runtime_task(Worker&, Node*);
  void _invoke_condition_task(Worker&, Node*, SmallVector<int>&);
  void _invoke_multi_condition_task(Worker&, Node*, SmallVector<int>&);
  void _process_dependent_async(Node*, tf::AsyncTask&, size_t&);
  void _process_exception(Worker&, Node*);
  void _update_cache(Worker&, Node*&, Node*);
  void _corun_graph(Worker&, Graph&, Topology*, NodeBase*);

  bool _wait_for_task(Worker&, Node*&);
  bool _invoke_subflow_task(Worker&, Node*);
  bool _invoke_module_task(Worker&, Node*);
  bool _invoke_adopted_module_task(Worker&, Node*);
  bool _invoke_module_task_impl(Worker&, Node*, Graph&);
  bool _invoke_async_task(Worker&, Node*);
  bool _invoke_dependent_async_task(Worker&, Node*);
  bool _invoke_runtime_task(Worker&, Node*);
  bool _invoke_runtime_task_impl(Worker&, Node*, std::function<void(Runtime&)>&);
  bool _invoke_runtime_task_impl(Worker&, Node*, std::function<void(Runtime&, bool)>&);

  size_t _set_up_graph(Graph&, Topology*, NodeBase*);
  
  template <typename P>
  void _corun_until(Worker&, P&&);

  template <std::input_iterator I>
  void _bulk_schedule(Worker&, I, size_t);

  template <std::input_iterator I>
  void _bulk_schedule(I, size_t);

  template <std::input_iterator I>
  void _bulk_spill(I, size_t);
  
  template <std::input_iterator I>
  void _bulk_spill_round_robin(I, size_t);

  template <size_t N>
  void _bulk_update_cache(Worker&, Node*&, Node*, std::array<Node*, N>&, size_t&);

  template <TaskParamsLike P, typename F>
  auto _async(P&&, F&&, Topology*, NodeBase*);

  template <TaskParamsLike P, typename F>
  void _silent_async(P&&, F&&, Topology*, NodeBase*);

  template <TaskParamsLike P, typename F, std::input_iterator I>
  auto _dependent_async(P&&, F&&, I, I, Topology*, NodeBase*);
  
  template <TaskParamsLike P, typename F, std::input_iterator I>
  auto _silent_dependent_async(P&&, F&&, I, I, Topology*, NodeBase*);
  
  template <typename... ArgsT>
  void _schedule_async_task(ArgsT&&...);
  
  template <std::input_iterator I, typename... ArgsT>
  AsyncTask _schedule_dependent_async_task(I, I, size_t, ArgsT&&...);
};

#ifndef DOXYGEN_GENERATING_OUTPUT

// Constructor
inline Executor::Executor(size_t N, std::shared_ptr<WorkerInterface> wif) :
  _workers  (N),
  _buffers  (std::bit_width(N)), // Empirically, we find that log2(N) performs best.
  _notifier (N) {

  // 0 workers => inline mode, run graphs on the caller.
  if(N == 0) {
    _inlined = true;
    return;
  }

  // If spawning N threads fails, shut down any created threads before 
  // rethrowing the exception.
#ifndef TF_DISABLE_EXCEPTION_HANDLING
  try {
#endif
    _spawn(N, std::move(wif));
#ifndef TF_DISABLE_EXCEPTION_HANDLING
  }
  catch(...) {
    _shutdown();
    std::rethrow_exception(std::current_exception());
  }
#endif

  // initialize the default observer if requested
  if(has_env(TF_ENABLE_PROFILER)) {
    TFProfManager::get()._manage(make_observer<TFProfObserver>());
  }
}

// Destructor
inline Executor::~Executor() {
  _shutdown();
}

// Function: _shutdown
inline void Executor::_shutdown() {

  // wait for all topologies to complete
  wait_for_all();

  // shut down the scheduler
  for(size_t i=0; i<_workers.size(); ++i) {
    _workers[i]._done.test_and_set(std::memory_order_relaxed);
  }
  
  _notifier.notify_all();
  
  // Only join the thread if it is joinable, as std::thread construction 
  // may fail and throw an exception.
  for(auto& w : _workers) {
    if(w._thread.joinable()) {
      w._thread.join();
    }
  }
}

// Function: num_workers
inline size_t Executor::num_workers() const noexcept {
  return _workers.size();
}

// Function: num_waiters
inline size_t Executor::num_waiters() const noexcept {
  return _notifier.num_waiters();
}

// Function: num_queues
inline size_t Executor::num_queues() const noexcept {
  return _workers.size() + _buffers.size();
}

// Function: num_topologies
inline size_t Executor::num_topologies() const {
  return _num_topologies.load(std::memory_order_relaxed);
}

// Function: this_worker
inline Worker* Executor::this_worker() {
  auto itr = _t2w.find(std::this_thread::get_id());
  return itr == _t2w.end() ? nullptr : itr->second;
}

// Function: this_worker_id
inline int Executor::this_worker_id() const {
  // in inline mode the calling thread is the sole worker 0.
  if(_inlined) {
    return 0;
  }
  auto i = _t2w.find(std::this_thread::get_id());
  return i == _t2w.end() ? -1 : static_cast<int>(i->second->_id);
}

// Procedure: _spawn
inline void Executor::_spawn(size_t N, std::shared_ptr<WorkerInterface> wif) {

  for(size_t id=0; id<N; ++id) {
    _workers[id]._thread = std::thread([&, id, wif] () {

      auto& worker = _workers[id];
  
      worker._id = id;
      worker._sticky_victim = id;
      worker._rdgen.seed(static_cast<uint32_t>(std::hash<std::thread::id>()(std::this_thread::get_id())));

      // before entering the work-stealing loop, call the scheduler prologue
      if(wif) {
        wif->scheduler_prologue(worker);
      }

      Node* t = nullptr;
      std::exception_ptr ptr = nullptr;

      // must use 1 as condition instead of !done because
      // the previous worker may stop while the following workers
      // are still preparing for entering the scheduling loop
#ifndef TF_DISABLE_EXCEPTION_HANDLING
      try {
#endif
        // work-stealing loop
        while(1) {

          // drains out the local queue first
          _exploit_task(worker, t);

          // steals and waits for tasks
          if(_wait_for_task(worker, t) == false) {
            break;
          }
        }

#ifndef TF_DISABLE_EXCEPTION_HANDLING
      } 
      catch(...) {
        ptr = std::current_exception();
      }
#endif
      
      // call the user-specified epilogue function
      if(wif) {
        wif->scheduler_epilogue(worker, ptr);
      }

    });
    
    // We avoid using thread-local storage to track the mapping between a thread
    // and its corresponding worker in an executor. On Windows, thread-local
    // storage can be unreliable in certain situations (see issue #727).
    //
    // Instead, we maintain a per-executor mapping from threads to workers.
    // This approach has an additional advantage: according to the C++ Standard,
    // std::thread::id uniquely identifies a thread object. Therefore, once the map
    // returns a valid worker, we can be certain that the worker belongs to this
    // executor. This eliminates the need for additional executor validation 
    // required by using thread-local storage.
    //
    // Example:
    //
    //   Worker* w = this_worker();
    //   // Using thread-local storage, we would need additional executor validation:
    //   if (w == nullptr || w->_executor != this) { /* caller is not a worker of this executor */ }
    //
    //   // Using per-executor mapping, it suffices to check:
    //   if (w == nullptr) { /* caller is not a worker of this executor */ }
    //
    _t2w.emplace(_workers[id]._thread.get_id(), &_workers[id]);
  }
}

// Function: _explore_task
inline bool Executor::_explore_task(Worker& w, Node*& t) {
 
  // Fast path: if no topologies are live, all queues are guaranteed empty
  // by the executor's invariant (num_topologies reaches zero only after all
  // nodes have been scheduled and their queues flushed). Skip the entire
  // steal loop and return immediately so the caller enters _wait_for_task
  // to sleep. relaxed ordering is sufficient — this is a hint, and any
  // missed update is caught safely by the 2PC guard in _wait_for_task.
  if(_num_topologies.load(std::memory_order_relaxed) == 0) {
    return true;
  }
 
  const size_t MAX_VICTIM = num_queues();  // guaranteed >= 2 by constructor
  const size_t MAX_STEALS = ((MAX_VICTIM + 1) << 1);
 
  // local aliases for steal protocol sentinels — these are properties of the
  // steal protocol, not of any specific queue type
  size_t num_steals = 0;
  size_t vtm = w._sticky_victim;
 
  while(true) {
 
    t = (vtm < _workers.size())
      ? _workers[vtm]._wsq.steal()
      : _buffers[vtm - _workers.size()].queue.steal();
 
    if(t) {
      w._sticky_victim = vtm;
      break;
    }
    
    // EMPTY: pick a new victim excluding self since our own queue is likely empty.
    // map [0, MAX_VICTIM-1) over [0, MAX_VICTIM) \ {w._id} — always safe since MAX_VICTIM >= 2.
    vtm = w._rdgen() % (MAX_VICTIM - 1);
    if(vtm >= w._id) vtm++;
 
    if(++num_steals > MAX_STEALS) {
      std::this_thread::yield();
      if(num_steals > 16 + MAX_STEALS) {
        break;
      }
    }
 
    if(w._done.test(std::memory_order_relaxed)) {
      return false;
    }
  }
 
  return true;
} 

/*
// Function: _explore_task
inline bool Executor::_explore_task(Worker& w, Node*& t) {

  // Fast path: if no topologies are live, all queues are guaranteed empty
  // by the executor's invariant (num_topologies reaches zero only after all
  // nodes have been scheduled and their queues flushed). Skip the entire
  // steal loop and return immediately so the caller enters _wait_for_task
  // to sleep. relaxed ordering is sufficient — this is a hint, and any
  // missed update is caught safely by the 2PC guard in _wait_for_task.
  if(_num_topologies.load(std::memory_order_relaxed) == 0) {
    return true;
  }

  //assert(!t);
  const size_t MAX_VICTIM = num_queues();
  const size_t MAX_STEALS = ((MAX_VICTIM + 1) << 1);

  size_t num_steals = 0;
  size_t vtm = w._sticky_victim;

  // Make the worker steal immediately from the assigned victim.
  while(true) {
    
    // If the worker's victim thread is within the worker pool, steal from the worker's queue.
    // Otherwise, steal from the buffer, adjusting the victim index based on the worker pool size.
    t = (vtm < _workers.size())
      ? _workers[vtm]._wsq.steal()
      : _buffers[vtm - _workers.size()].queue.steal();

    if(t) {
      w._sticky_victim = vtm;
      break;
    }

    // Increment the steal count, and if it exceeds MAX_STEALS, yield the thread.
    // If the number of empty steals reaches MAX_STEALS, exit the loop.
    if (++num_steals > MAX_STEALS) {
      std::this_thread::yield();
      if(num_steals > 16 + MAX_STEALS) {
        break;
      }
    }

    if(w._done.test(std::memory_order_relaxed)) {
      return false;
    } 

    // Randomely generate a next victim.
    vtm = w._rdgen() % MAX_VICTIM;
  } 
  return true;
}
*/

// Procedure: _exploit_task
inline void Executor::_exploit_task(Worker& w, Node*& t) {
  while(t) {
    _invoke(w, t);
    t = w._wsq.pop();
  }
}

// Function: _wait_for_task
inline bool Executor::_wait_for_task(Worker& w, Node*& t) {

  explore_task:

  if(_explore_task(w, t) == false) {
    return false;
  }
  
  // Go exploit the task if we successfully steal one.
  if(t) {
    return true;
  }

  // Entering the 2PC guard as all queues are likely empty after many stealing attempts.
  _notifier.prepare_wait(w._id);

  // Fast path: if no topologies are live, all queues are guaranteed empty.
  // Skip the O(N) buffer and worker queue scans and go directly to sleep.
  // This is safe because prepare_wait has already been called — any notify
  // that arrives after this check but before commit_wait will be caught by
  // the 2PC guarantee of the notifier.
  if(_num_topologies.load(std::memory_order_relaxed) == 0) {
    // still check done flag before committing to sleep
    if(w._done.test(std::memory_order_relaxed)) {
      _notifier.cancel_wait(w._id);
      return false;
    }
    _notifier.commit_wait(w._id);
    goto explore_task;
  }
  
  // Condition #1: buffers should be empty
  for(size_t b=0; b<_buffers.size(); ++b) {
    if(!_buffers[b].queue.empty()) {
      _notifier.cancel_wait(w._id);
      w._sticky_victim = b + _workers.size();
      goto explore_task;
    }
  }
  
  // Condition #2: worker queues should be empty
  // Note: We need to use index-based looping to avoid data race with _spawn
  // which initializes other worker data structure at the same time.
  // Also, due to the property of a work-stealing queue, we don't need to check 
  // this worker's work-stealing queue.
  for(size_t k=0; k<_workers.size()-1; ++k) {
    if(size_t vtm = k + (k >= w._id); !_workers[vtm]._wsq.empty()) {
      _notifier.cancel_wait(w._id);
      w._sticky_victim = vtm;
      goto explore_task;
    }
  }
  
  // Condition #3: worker should be alive
  if(w._done.test(std::memory_order_relaxed)) {
    _notifier.cancel_wait(w._id);
    return false;
  }
  
  // Now I really need to relinquish myself to others.
  _notifier.commit_wait(w._id);
  goto explore_task;
}

// Function: make_observer
template<typename Observer, typename... ArgsT>
std::shared_ptr<Observer> Executor::make_observer(ArgsT&&... args) {

  static_assert(
    std::is_base_of_v<ObserverInterface, Observer>,
    "Observer must be derived from ObserverInterface"
  );

  // use a local variable to mimic the constructor
  auto ptr = std::make_shared<Observer>(std::forward<ArgsT>(args)...);

  ptr->set_up(_workers.size());

  _observers.emplace(std::static_pointer_cast<ObserverInterface>(ptr));

  return ptr;
}

// Procedure: remove_observer
template <typename Observer>
void Executor::remove_observer(std::shared_ptr<Observer> ptr) {

  static_assert(
    std::is_base_of_v<ObserverInterface, Observer>,
    "Observer must be derived from ObserverInterface"
  );

  _observers.erase(std::static_pointer_cast<ObserverInterface>(ptr));
}

// Function: num_observers
inline size_t Executor::num_observers() const noexcept {
  return _observers.size();
}

// Procedure: _spill
inline void Executor::_spill(Node* item) {
  // Since pointers are aligned to 8 bytes, we perform a simple hash to avoid 
  // contention caused by hashing to the same slot.
  auto b = (reinterpret_cast<uintptr_t>(item) >> 16) % _buffers.size();
  std::scoped_lock lock(_buffers[b].mutex);
  _buffers[b].queue.push(item);
}

// Procedure: _bulk_spill (single batch to one buffer)
// Uses Knuth multiplicative hash on the first pointer to select a buffer,
// providing better bit diffusion than the shift-based approach, especially
// when the allocator returns pointers with regular low-bit patterns.
template <std::input_iterator I>
void Executor::_bulk_spill(I first, size_t N) {
  //assert(N != 0);
  auto b = ((reinterpret_cast<uintptr_t>(*first) * 2654435761ULL) >> 32) % _buffers.size();
  std::scoped_lock lock(_buffers[b].mutex);
  _buffers[b].queue.bulk_push(first, N);
}

// Procedure: _bulk_spill
// Distributes a batch of N spilled nodes across all buffers in round-robin
// order starting from a hash of the first node's pointer. Each buffer's lock
// is held only for its chunk, reducing contention compared to sending the
// entire batch to a single buffer.
template <std::input_iterator I>
void Executor::_bulk_spill_round_robin(I first, size_t N) {

  // assert(N != 0);
  const size_t B     = _buffers.size();
  const size_t start = ((reinterpret_cast<uintptr_t>(*first) * 2654435761ULL) >> 32) % B;
  const size_t per_buf = (N + B - 1) / B;
  size_t remaining = N;
  for(size_t i = 0; i < B && remaining > 0; ++i) {
    size_t b     = (start + i) % B;
    size_t chunk = std::min(per_buf, remaining);
    {
      std::scoped_lock lock(_buffers[b].mutex);
      _buffers[b].queue.bulk_push(first, chunk);
    }
    // terminates early via remaining > 0, so we don't acquire unnecessary locks on empty chunks.
    remaining -= chunk;
  }
}

// Procedure: _schedule
inline void Executor::_schedule(Worker& worker, Node* node) {
  // starting at v3.5 we do not use any complicated notification mechanism 
  // as the experimental result has shown no significant advantage.
  if(worker._wsq.try_push(node) == false) {
    _spill(node);
  }
  _notifier.notify_one();
}

// Procedure: _schedule
inline void Executor::_schedule(Node* node) {
  _spill(node);
  _notifier.notify_one();
}

// Procedure: _schedule
template <std::input_iterator I>
void Executor::_bulk_schedule(Worker& worker, I first, size_t num_nodes) {

  if(num_nodes == 0) {
    return;
  }

  // NOTE: We cannot use first/last in the for-loop (e.g., for(; first != last; ++first)).
  // This is because when a node v is inserted into the queue, v can run and finish 
  // immediately. If v is the last node in the graph, it will tear down the parent task vector
  // which cause the last ++first to fail. This problem is specific to MSVC which has a stricter
  // iterator implementation in std::vector than GCC/Clang.
  if(auto n = worker._wsq.try_bulk_push(first, num_nodes); n != num_nodes) {
    _bulk_spill(first, num_nodes - n);
  }
  _notifier.notify_n(num_nodes);
    
  // notify first before spilling to hopefully wake up workers earlier 
  // however, the experiment does not show any benefit for doing this.
  //auto n = worker._wsq.try_bulk_push(first, num_nodes);
  //_notifier.notify_n(n);
  //_bulk_schedule(first + n, num_nodes - n);
}

// Procedure: _schedule
template <std::input_iterator I>
inline void Executor::_bulk_schedule(I first, size_t num_nodes) {
  
  if(num_nodes == 0) {
    return;
  }
  
  // NOTE: We cannot use first/last in the for-loop (e.g., for(; first != last; ++first)).
  // This is because when a node v is inserted into the queue, v can run and finish 
  // immediately. If v is the last node in the graph, it will tear down the parent task vector
  // which cause the last ++first to fail. This problem is specific to MSVC which has a stricter
  // iterator implementation in std::vector than GCC/Clang.
  _bulk_spill(first, num_nodes);
  _notifier.notify_n(num_nodes);
}

// Function: _update_cache
TF_FORCE_INLINE void Executor::_update_cache(Worker& worker, Node*& cache, Node* node) {
  if(cache) {
    _schedule(worker, cache);
  }
  cache = node;
}

// Function: _bulk_update_cache
template <size_t N>
TF_FORCE_INLINE void Executor::_bulk_update_cache(
  Worker& worker, Node*& cache, Node* node, std::array<Node*, N>& array, size_t& n
) {
  // experimental results show no benefit of using bulk_update_cache
  if(cache) {
    array[n++] = cache;
    if(n == N) {
      _bulk_schedule(worker, array, n);
      n = 0;
    }
  }
  cache = node;
}
  
// Procedure: _invoke
inline void Executor::_invoke(Worker& worker, Node* node) {

  #define TF_INVOKE_CONTINUATION()  \
  if (cache) {                      \
    node = cache;                   \
    goto begin_invoke;              \
  }

  begin_invoke:

  Node* cache {nullptr};
  
  // if this is the second invoke due to preemption, directly jump to invoke task
  if(node->_nstate & NSTATE::PREEMPTED) {
    goto invoke_task;
  }

  // If the work has been cancelled, there is no need to continue.
  // Here, we do tear_down_invoke since async tasks may also get cancelled where
  // we need to recycle the node.
  if(node->_is_parent_cancelled()) {
    _tear_down_invoke(worker, node, cache);
    TF_INVOKE_CONTINUATION();
    return;
  }

  // if acquiring semaphore(s) exists, acquire them first
  if(node->_semaphores && !node->_semaphores->to_acquire.empty()) {
    SmallVector<Node*> waiters;
    if(!node->_acquire_all(waiters)) {
      _bulk_schedule(worker, waiters.begin(), waiters.size());
      return;
    }
  }
  
  invoke_task:
  
  SmallVector<int> conds;

  // switch is faster than nested if-else due to jump table
  switch(node->_handle.index()) {
    // static task
    case Node::STATIC:{
      _invoke_static_task(worker, node);
    }
    break;
    
    // runtime task
    case Node::RUNTIME:{
      if(_invoke_runtime_task(worker, node)) {
        return;
      }
    }
    break;
    
    // non-preemptive runtime task
    case Node::NONPREEMPTIVE_RUNTIME:{
      _invoke_nonpreemptive_runtime_task(worker, node);
    }
    break;

    // subflow task
    case Node::SUBFLOW: {
      if(_invoke_subflow_task(worker, node)) {
        return;
      }
    }
    break;

    // condition task
    case Node::CONDITION: {
      _invoke_condition_task(worker, node, conds);
    }
    break;

    // multi-condition task
    case Node::MULTI_CONDITION: {
      _invoke_multi_condition_task(worker, node, conds);
    }
    break;

    // module task
    case Node::MODULE: {
      if(_invoke_module_task(worker, node)) {
        return;
      }
    }
    break;
    
    // adopted module task
    case Node::ADOPTED_MODULE: {
      if(_invoke_adopted_module_task(worker, node)) {
        return;
      }
    }
    break;

    // async task
    case Node::ASYNC: {
      if(_invoke_async_task(worker, node)) {
        return;
      }
      _tear_down_async(worker, node, cache);
      TF_INVOKE_CONTINUATION();
      return;
    }
    break;

    // dependent async task
    case Node::DEPENDENT_ASYNC: {
      if(_invoke_dependent_async_task(worker, node)) {
        return;
      }
      _tear_down_dependent_async(worker, node, cache);
      TF_INVOKE_CONTINUATION();
      return;
    }
    break;

    // monostate (placeholder)
    default:
    break;
  }

  // if releasing semaphores exist, release them
  if(node->_semaphores && !node->_semaphores->to_release.empty()) {
    SmallVector<Node*> waiters;
    node->_release_all(waiters);
    _bulk_schedule(worker, waiters.begin(), waiters.size());
  }

  // Reset the join counter with strong dependencies to support cycles.
  // + We must do this before scheduling the successors to avoid race
  //   condition on _predecessors.
  // + We must use fetch_add instead of direct assigning
  //   because the user-level call on "invoke" may explicitly schedule 
  //   this task again (e.g., pipeline) which can access the join_counter.
  node->_join_counter.fetch_add(
    node->_nstate & NSTATE::STRONG_DEPENDENCIES_MASK, std::memory_order_relaxed
  );

  // Invoke the task based on the corresponding type
  switch(node->_handle.index()) {

    // condition and multi-condition tasks
    case Node::CONDITION:
    case Node::MULTI_CONDITION: {
      for(auto cond : conds) {
        if(cond >= 0 && static_cast<size_t>(cond) < node->_num_successors) {
          auto s = node->_edges[cond]; 
          // zeroing the join counter for invariant
          s->_join_counter.store(0, std::memory_order_relaxed);
          node->_parent->_join_counter.fetch_add(1, std::memory_order_relaxed);
          _update_cache(worker, cache, s);
        }
      }
    }
    break;

    // non-condition task
    default: {
      for(size_t i=0; i<node->_num_successors; ++i) {
        if(auto s = node->_edges[i]; s->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
          node->_parent->_join_counter.fetch_add(1, std::memory_order_relaxed);
          _update_cache(worker, cache, s);
        }
      }
    }
    break;
  }

  // clean up the node after execution
  _tear_down_nonasync(worker, node, cache);
  TF_INVOKE_CONTINUATION();
}

// Procedure: _tear_down_nonasync
inline void Executor::_tear_down_nonasync(Worker& worker, Node* node, Node*& cache) {

  // we must check parent first before subtracting the join counter,
  // or it can introduce data race
  if(auto parent = node->_parent; parent == node->_topology) {
    if(parent->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      _tear_down_topology(worker, node->_topology, cache);
    }
  }
  else {  
    // needs to fetch every data before join counter becomes zero at which
    // the node may be deleted
    auto state = parent->_nstate;
    if(parent->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      // this task is spawned from a preempted parent, so we need to resume it
      if(state & NSTATE::PREEMPTED) {
        _update_cache(worker, cache, static_cast<Node*>(parent));
      }
    }
  }
}

// Procedure: _tear_down_invoke
inline void Executor::_tear_down_invoke(Worker& worker, Node* node, Node*& cache) {
  switch(node->_handle.index()) {
    case Node::ASYNC:
      _tear_down_async(worker, node, cache);
    break;

    case Node::DEPENDENT_ASYNC:
      _tear_down_dependent_async(worker, node, cache);
    break;

    default:
      _tear_down_nonasync(worker, node, cache);
    break;
  }
}

// Procedure: _observer_prologue
inline void Executor::_observer_prologue(Worker& worker, Node* node) {
  for(auto& observer : _observers) {
    observer->on_entry(WorkerView(worker), TaskView(*node));
  }
}

// Procedure: _observer_epilogue
inline void Executor::_observer_epilogue(Worker& worker, Node* node) {
  for(auto& observer : _observers) {
    observer->on_exit(WorkerView(worker), TaskView(*node));
  }
}

// Procedure: _process_exception
inline void Executor::_process_exception(Worker&, Node* node) {

  // Finds the anchor and mark the entire path with exception, 
  // so recursive tasks can be cancelled properly.
  // Since exception can come from asynchronous task (with runtime), the node itself can be anchored.
  NodeBase* ea = node;     // explicit anchor
  NodeBase* ia = nullptr;  // implicit anchor
  
  while(ea && (ea->_estate.load(std::memory_order_relaxed) & ESTATE::EXPLICITLY_ANCHORED) == 0) {
    ea->_estate.fetch_or(ESTATE::EXCEPTION, std::memory_order_relaxed);
    // we only want the inner-most implicit anchor
    if(ia == nullptr && (ea->_nstate & NSTATE::IMPLICITLY_ANCHORED)) {
      ia = ea;
    }
    ea = ea->_parent;
  }
  
  // flag used to ensure execution is caught in a thread-safe manner
  constexpr static auto flag = ESTATE::EXCEPTION | ESTATE::CAUGHT;

  // The exception occurs under a blocking call (e.g., corun, join).
  if(ea) {
    // multiple tasks may throw, and we only take the first thrown exception
    if((ea->_estate.fetch_or(flag, std::memory_order_relaxed) & ESTATE::CAUGHT) == 0) {
      ea->_exception_ptr = std::current_exception();
      return;
    }
  }
  // Implicit anchor has the lowest priority
  else if(ia){
    if((ia->_estate.fetch_or(flag, std::memory_order_relaxed) & ESTATE::CAUGHT) == 0) {
      ia->_exception_ptr = std::current_exception();
      return;
    }
  }
  
  // For now, we simply store the exception in this node; this can happen in an 
  // execution that does not have any external control to capture the exception,
  // such as silent async task without any parent.
  node->_exception_ptr = std::current_exception();
}

// Procedure: _invoke_static_task
inline void Executor::_invoke_static_task(Worker& worker, Node* node) {
  _observer_prologue(worker, node);
  TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
    std::get_if<Node::Static>(&node->_handle)->work();
  });
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_subflow_task
inline bool Executor::_invoke_subflow_task(Worker& worker, Node* node) {
    
  auto& h = *std::get_if<Node::Subflow>(&node->_handle);
  auto& g = h.subgraph;

  if((node->_nstate & NSTATE::PREEMPTED) == 0) {
    
    // set up the subflow
    Subflow sf(*this, worker, node, g);

    // invoke the subflow callable
    _observer_prologue(worker, node);
    TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
      h.work(sf);
    });
    _observer_epilogue(worker, node);
    
    // spawn the subflow if it is joinable and its graph is non-empty
    // implicit join is faster than Subflow::join as it does not involve corun
    if(sf.joinable() && !g.empty()) {

      // signal the executor to preempt this node
      node->_nstate |= NSTATE::PREEMPTED;

      // set up and schedule the graph
      _schedule_graph(worker, g, node->_topology, node);
      return true;
    }
  }
  else {
    node->_nstate &= ~NSTATE::PREEMPTED;
  }

  // The subflow has finished or joined.
  // By default, we clear the subflow storage as applications can perform recursive
  // subflow tasking which accumulates a huge amount of memory overhead, hampering 
  // the performance.
  if((node->_nstate & NSTATE::RETAIN_SUBFLOW) == 0) {
    g.clear();
  }

  return false;
}

// Procedure: _invoke_condition_task
inline void Executor::_invoke_condition_task(
  Worker& worker, Node* node, SmallVector<int>& conds
) {
  _observer_prologue(worker, node);
  TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
    auto& work = std::get_if<Node::Condition>(&node->_handle)->work;
    conds = { work() };
  });
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_multi_condition_task
inline void Executor::_invoke_multi_condition_task(
  Worker& worker, Node* node, SmallVector<int>& conds
) {
  _observer_prologue(worker, node);
  TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
    conds = std::get_if<Node::MultiCondition>(&node->_handle)->work();
  });
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_module_task
inline bool Executor::_invoke_module_task(Worker& w, Node* node) {
  return _invoke_module_task_impl(w, node, std::get_if<Node::Module>(&node->_handle)->graph);  
}

// Procedure: _invoke_adopted_module_task
inline bool Executor::_invoke_adopted_module_task(Worker& w, Node* node) {
  return _invoke_module_task_impl(w, node, std::get_if<Node::AdoptedModule>(&node->_handle)->graph);  
}

// Procedure: _invoke_module_task_impl
inline bool Executor::_invoke_module_task_impl(Worker& w, Node* node, Graph& graph) {

  // No need to do anything for empty graph
  if(graph.empty()) {
    return false;
  }

  // first entry - not spawned yet
  if((node->_nstate & NSTATE::PREEMPTED) == 0) {
    // signal the executor to preempt this node
    node->_nstate |= NSTATE::PREEMPTED;
    _schedule_graph(w, graph, node->_topology, node);
    return true;
  }

  // second entry - already spawned
  node->_nstate &= ~NSTATE::PREEMPTED;

  return false;
}


// Procedure: _invoke_async_task
inline bool Executor::_invoke_async_task(Worker& worker, Node* node) {
  auto& work = std::get_if<Node::Async>(&node->_handle)->work;
  switch(work.index()) {
    // void()
    case 0:
      _observer_prologue(worker, node);
      TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
        std::get_if<0>(&work)->operator()();
      });
      _observer_epilogue(worker, node);
    break;
    
    // void(Runtime&)
    case 1:
      if(_invoke_runtime_task_impl(worker, node, *std::get_if<1>(&work))) {
        return true;
      }
    break;
    
    // void(Runtime&, bool)
    case 2:
      if(_invoke_runtime_task_impl(worker, node, *std::get_if<2>(&work))) {
        return true;
      }
    break;
  }

  return false;
}

// Procedure: _invoke_dependent_async_task
inline bool Executor::_invoke_dependent_async_task(Worker& worker, Node* node) {
  auto& work = std::get_if<Node::DependentAsync>(&node->_handle)->work;
  switch(work.index()) {
    // void()
    case 0:
      _observer_prologue(worker, node);
      TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
        std::get_if<0>(&work)->operator()();
      });
      _observer_epilogue(worker, node);
    break;
    
    // void(Runtime&) - silent async
    case 1:
      if(_invoke_runtime_task_impl(worker, node, *std::get_if<1>(&work))) {
        return true;
      }
    break;

    // void(Runtime&, bool) - async
    case 2:
      if(_invoke_runtime_task_impl(worker, node, *std::get_if<2>(&work))) {
        return true;
      }
    break;
  }
  return false;
}

// Function: run
inline tf::Future<void> Executor::run(Taskflow& f) {
  return run_n(f, 1, [](){});
}

// Function: run
inline tf::Future<void> Executor::run(Taskflow&& f) {
  return run_n(std::move(f), 1, [](){});
}

// Function: run
template <typename C>
tf::Future<void> Executor::run(Taskflow& f, C&& c) {
  return run_n(f, 1, std::forward<C>(c));
}

// Function: run
template <typename C>
tf::Future<void> Executor::run(Taskflow&& f, C&& c) {
  return run_n(std::move(f), 1, std::forward<C>(c));
}

// Function: run_n
inline tf::Future<void> Executor::run_n(Taskflow& f, size_t repeat) {
  return run_n(f, repeat, [](){});
}

// Function: run_n
inline tf::Future<void> Executor::run_n(Taskflow&& f, size_t repeat) {
  return run_n(std::move(f), repeat, [](){});
}

// Function: run_n
template <typename C>
tf::Future<void> Executor::run_n(Taskflow& f, size_t repeat, C&& c) {
  return run_until(
    f, [repeat]() mutable { return repeat-- == 0; }, std::forward<C>(c)
  );
}

// Function: run_n
template <typename C>
tf::Future<void> Executor::run_n(Taskflow&& f, size_t repeat, C&& c) {
  return run_until(
    std::move(f), [repeat]() mutable { return repeat-- == 0; }, std::forward<C>(c)
  );
}

// Function: run_until
template<typename P>
tf::Future<void> Executor::run_until(Taskflow& f, P&& pred) {
  return run_until(f, std::forward<P>(pred), [](){});
}

// Function: run_until
template<typename P>
tf::Future<void> Executor::run_until(Taskflow&& f, P&& pred) {
  return run_until(std::move(f), std::forward<P>(pred), [](){});
}

// Function: run_until
template <typename P, typename C>
tf::Future<void> Executor::run_until(Taskflow& f, P&& p, C&& c) {

  // inline (0-worker) mode runs the graph synchronously on
  // the calling thread and returns an already-satisfied future.
  if(_inlined) {
    std::exception_ptr eptr;
    try {
      while(!p()) {
        _run_inline(f._graph);
      }
    }
    catch(...) {
      eptr = std::current_exception();
    }
    c();
    std::promise<void> promise;
    if(eptr) {
      promise.set_exception(eptr);
    }
    else {
      promise.set_value();
    }
    return tf::Future<void>(promise.get_future());
  }

  // No need to create a real topology but returns an dummy future for invariant.
  if(f.empty() || p()) {
    c();
    std::promise<void> promise;
    promise.set_value();
    return tf::Future<void>(promise.get_future());
  }
  
  _increment_topology();

  // create a topology for this run
  auto t = std::make_shared<Topology>(f, std::forward<P>(p), std::forward<C>(c));
  //auto t = std::make_shared<DerivedTopology<P, C>>(f, std::forward<P>(p), std::forward<C>(c));

  // need to create future before the topology got torn down quickly
  tf::Future<void> future(t->_promise.get_future(), t);

  // modifying topology needs to be protected under the lock
  if(f._fetch_enqueue(t) == 0) {
    _set_up_topology(this_worker(), t.get());
  }

  return future;
}



// Function: corun_until
template <typename P>
void Executor::corun_until(P&& predicate) {
  
  Worker* w = this_worker();
  if(w == nullptr) {
    TF_THROW("corun_until must be called by a worker of the executor");
  }

  _corun_until(*w, std::forward<P>(predicate));
}

/*
// Function: _corun_until
template <typename P>
void Executor::_corun_until(Worker& w, P&& stop_predicate) {
 
  const size_t MAX_VICTIM = num_queues();
  const size_t MAX_STEALS = ((MAX_VICTIM + 1) << 1);
 
  bool stop = false;
 
  while(!stop && !(stop = stop_predicate())) {
 
    // try local queue first — only one task at a time to avoid deep
    // recursive corun calls causing stack overflow
    if(auto t = w._wsq.pop(); t) {
      _invoke(w, t);
      continue;
    }
 
    // local queue empty: steal from others until stop_predicate or stolen.
    // stop is set by the inner loop condition so when predicate becomes true
    // the outer loop exits immediately without calling stop_predicate again.
    size_t num_steals = 0;
    size_t vtm = w._sticky_victim;
 
    while(!(stop = stop_predicate())) {
 
      auto t = (vtm < _workers.size())
        ? _workers[vtm]._wsq.steal()
        : _buffers[vtm - _workers.size()].queue.steal();
 
      if(t) {
        // STOLEN: invoke task then return to outer loop to re-check
        // local queue and stop_predicate
        _invoke(w, t);
        w._sticky_victim = vtm;
        break;
      }
 
      // pick a new victim excluding self
      vtm = w._rdgen() % (MAX_VICTIM - 1);
      if(vtm >= w._id) vtm++;
 
      if(++num_steals > MAX_STEALS) {
        // unlike _explore_task we cannot sleep here — the calling worker
        // is blocked inside a task and must keep making progress to avoid
        // deadlock. yield to let other threads run and make progress.
        std::this_thread::yield();
      }
    }
  }
}*/
 
// Function: _corun_until
template <typename P>
void Executor::_corun_until(Worker& w, P&& stop_predicate) {

  const size_t MAX_VICTIM = num_queues();
  const size_t MAX_STEALS = ((MAX_VICTIM + 1) << 1);
    
  exploit:

  while(!stop_predicate()) {
    
    // here we don't do while-loop to drain out the local queue as it can
    // potentially enter a very deep recursive corun, cuasing stack overflow
    if(auto t = w._wsq.pop(); t) {
      _invoke(w, t);
    }
    else {
      size_t num_steals = 0;
      size_t vtm = w._sticky_victim;

      explore:

      t = (vtm < _workers.size()) 
        ? _workers[vtm]._wsq.steal()
        : _buffers[vtm-_workers.size()].queue.steal();

      if(t) {
        _invoke(w, t);
        w._sticky_victim = vtm;
        goto exploit;
      }
      else if(!stop_predicate()) {
        if(++num_steals > MAX_STEALS) {
          std::this_thread::yield();
        }
        vtm = w._rdgen() % MAX_VICTIM;
        goto explore;
      }
      else {
        break;
      }
    }
  }
}

// Function: corun
template <typename T>
void Executor::corun(T& target) {

  // inline mode has no workers; run the graph synchronously.
  if(_inlined) {
    _run_inline(retrieve_graph(target));
    return;
  }

  Worker* w = this_worker();
  if(w == nullptr) {
    TF_THROW("corun must be called by a worker of the executor");
  }

  NodeBase anchor;
  _corun_graph(*w, retrieve_graph(target), nullptr, &anchor);
}


// Procedure: _run_inline
// Executes a fully-built static graph on the calling thread in topological
// order, mirroring the executor's scheduling rules: a node runs once its
// strong (non-conditioner) predecessors are met; a condition node runs its
// int() work and activates only the selected successor.  Handles Static,
// Module (composed_of, recursed), Condition/MultiCondition and placeholders.
// Assumes the graph is complete before run (no tasks added during execution)
// and no subflow/runtime tasks
inline void Executor::_run_inline(Graph& graph) {
  if(graph.empty()) {
    return;
  }

  std::unordered_map<Node*, size_t> remaining;  // unmet strong predecessors
  std::unordered_set<Node*> pushed;
  remaining.reserve(graph.size());
  std::vector<Node*> ready;

  for(Node* n : graph) {
    size_t strong = 0;
    for(size_t i = n->_num_successors; i < n->_edges.size(); ++i) {
      strong += !n->_edges[i]->_is_conditioner();
    }
    remaining.emplace(n, strong);
    if(n->num_predecessors() == 0) {
      ready.push_back(n);
      pushed.insert(n);
    }
  }

  // A condition forces its chosen successor to run regardless of strong deps.
  auto activate = [&](Node* s) {
    remaining[s] = 0;
    if(pushed.insert(s).second) {
      ready.push_back(s);
    }
  };

  while(!ready.empty()) {
    Node* n = ready.back();
    ready.pop_back();

    switch(n->_handle.index()) {
      case Node::STATIC:
        std::get_if<Node::Static>(&n->_handle)->work();
        break;
      case Node::MODULE:
        _run_inline(std::get_if<Node::Module>(&n->_handle)->graph);
        break;
      case Node::CONDITION: {
        int idx = std::get_if<Node::Condition>(&n->_handle)->work();
        if(idx >= 0 && static_cast<size_t>(idx) < n->_num_successors) {
          activate(n->_edges[static_cast<size_t>(idx)]);
        }
        continue;  // conditioners propagate only to the chosen successor
      }
      case Node::MULTI_CONDITION: {
        for(int idx : std::get_if<Node::MultiCondition>(&n->_handle)->work()) {
          if(idx >= 0 && static_cast<size_t>(idx) < n->_num_successors) {
            activate(n->_edges[static_cast<size_t>(idx)]);
          }
        }
        continue;
      }
      default:
        break;  // placeholder / pure join node: nothing to execute
    }

    for(size_t i = 0; i < n->_num_successors; ++i) {
      Node* s = n->_edges[i];
      if(auto it = remaining.find(s);
         it != remaining.end() && it->second > 0 && --(it->second) == 0) {
        if(pushed.insert(s).second) {
          ready.push_back(s);
        }
      }
    }
  }
}

// Procedure: _corun_graph
inline void Executor::_corun_graph(Worker& w, Graph& g, Topology* tpg, NodeBase* p) {

  // empty graph
  if(g.empty()) {
    return;
  }
  
  // anchor this parent as the blocking point
  {
    ExplicitAnchorGuard anchor(p);
    _schedule_graph(w, g, tpg, p);
    _corun_until(w, [p] () -> bool { 
      return p->_join_counter.load(std::memory_order_acquire) == 0; }
    );
  }

  // rethrow the exception to the caller
  p->_rethrow_exception();
}

// Procedure: _increment_topology
inline void Executor::_increment_topology() {
  _num_topologies.fetch_add(1, std::memory_order_relaxed);
}

// Procedure: _decrement_topology
inline void Executor::_decrement_topology() {
  if(_num_topologies.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    _num_topologies.notify_all();
  }
}

// Procedure: wait_for_all
inline void Executor::wait_for_all() {
  size_t n = _num_topologies.load(std::memory_order_acquire);
  while(n != 0) {
    _num_topologies.wait(n, std::memory_order_acquire);
    n = _num_topologies.load(std::memory_order_acquire);
  }
}
  
// Function: _schedule_graph
inline void Executor::_schedule_graph(
  Worker& worker, Graph& graph, Topology* tpg, NodeBase* parent
) {
  size_t num_srcs = _set_up_graph(graph, tpg, parent);
  parent->_join_counter.fetch_add(num_srcs, std::memory_order_relaxed);
  _bulk_schedule(worker, graph.begin(), num_srcs);
}

// Function: _set_up_topology
inline void Executor::_set_up_topology(Worker* w, Topology* tpg) {
  // ---- under taskflow lock ----
  auto& g = tpg->_taskflow._graph;
  size_t num_srcs = _set_up_graph(g, tpg, tpg);
  tpg->_join_counter.store(num_srcs, std::memory_order_relaxed);
  w ? _bulk_schedule(*w, g.begin(), num_srcs) : _bulk_schedule(g.begin(), num_srcs);
}

// Function: _set_up_graph
inline size_t Executor::_set_up_graph(Graph& graph, Topology* tpg, NodeBase* parent) {

  auto first = graph.begin();
  auto last  = graph.end();
  auto send  = first;
  for(; first != last; ++first) {

    auto node = *first;
    node->_topology = tpg;
    node->_parent = parent;
    node->_nstate = NSTATE::NONE;
    node->_estate.store(ESTATE::NONE, std::memory_order_relaxed);
    node->_set_up_join_counter();
    node->_exception_ptr = nullptr;

    // move source to the first partition
    // root, root, root, v1, v2, v3, v4, ...
    if(node->num_predecessors() == 0) {
      std::iter_swap(send++, first);
    }
  }
  return send - graph.begin();
}

// Function: _tear_down_topology
inline void Executor::_tear_down_topology(Worker& worker, Topology* tpg, Node*& cache) {

  auto &f = tpg->_taskflow;

  //assert(&tpg == &(f._topologies.front()));

  // case 1: we still need to run the topology again
  //if(!tpg->_exception_ptr && !tpg->cancelled() && !tpg->predicate()) {
  if(!tpg->cancelled() && !tpg->_predicate()) {
    //assert(tpg->_join_counter == 0);
    //std::lock_guard<std::mutex> lock(f._mutex);
    _schedule_graph(worker, tpg->_taskflow._graph, tpg, tpg);
  }
  // case 2: the final run of this topology
  else {

    // invoke the callback after each run
    tpg->_on_finish();

    // there is another topologies to run
    if(std::unique_lock<std::mutex> lock(f._mutex); f._topologies.size()>1) {

      auto fetched_tpg {std::move(f._topologies.front())};
      //assert(fetched_tpg.get() == tpg);

      f._topologies.pop();
      tpg = f._topologies.front().get();

      lock.unlock();
      
      // Soon after we carry out the promise, the associate taskflow may got destroyed
      // from the user side, and we should never tough it again.
      fetched_tpg->_carry_out_promise();

      // decrement the topology
      _decrement_topology();

      _schedule_graph(worker, tpg->_taskflow._graph, tpg, tpg);
    }
    else {
      //assert(f._topologies.size() == 1);

      auto fetched_tpg {std::move(f._topologies.front())};
      //assert(fetched_tpg.get() == tpg);

      f._topologies.pop();

      lock.unlock();
      
      // Soon after we carry out the promise, the associate taskflow may got destroyed
      // from the user side, and we should never tough it again.
      fetched_tpg->_carry_out_promise();

      _decrement_topology();
      
      // remove the parent that owns the moved taskflow so the storage can be freed
      if(auto parent = fetched_tpg->_parent; parent) {
        //auto state = parent->_nstate;
        if(parent->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
          // this async is spawned from a preempted parent, so we need to resume it
          //if(state & NSTATE::PREEMPTED) { 
            _update_cache(worker, cache, static_cast<Node*>(parent));
          //}
        }
      }
    }
  }
}

// ############################################################################
// Forward Declaration: Subflow
// ############################################################################

inline void Subflow::join() {

  if(!joinable()) {
    TF_THROW("subflow already joined");
  }
    
  _executor._corun_graph(_worker, _graph, _node->_topology, _node);
  
  // join here since corun graph may throw exception
  _node->_nstate |= NSTATE::JOINED_SUBFLOW;
}

#endif




}  // end of namespace tf -----------------------------------------------------
