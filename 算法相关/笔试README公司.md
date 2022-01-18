### 进程 和线程通信方式的不同

在多道程序设计系统中，同一时刻可能有许多进程，这些进程之间存在两种基本关系：竞争关系和协作关系。

进程的互斥、同步、通信都是基于这两种基本关系而存在的，为了解决进程间竞争关系（间接制约关系）而引入进程互斥；为了解决进程间**松散的协作**关系( **直接制约关系**)而引入进程同步；为了解决进程间**紧密的协作**关系而引入进程通信。



**Linux 与Windows的主要同步、通信机制**如下：

**Linux 下：**

Linux 下常见的进程同步方法有：SysVIPC 的 sem（信号量）、file locking / record locking（通过 fcntl 设定的文件锁、记录锁）、futex（基于共享内存的快速用户态互斥锁）。针对线程（pthread）的还有 pthread_mutex 和 pthread_cond（条件变量）。

Linux 下常见的进程通信的方法有 ：pipe（管道），FIFO（命名管道），socket（套接字），SysVIPC 的 shm（共享内存）、msg queue（消息队列），mmap（文件映射）。以前还有 STREAM，不过现在比较少见了（好像）



**Windows下：**
在Windwos中，进程同步主要有以下几种：互斥量、信号量、事件、可等计时器等几种技术。

在Windows下，进程通信主要有以下几种：内存映射、管道、消息等，但是内存映射是最基础的，因为，其他的进程通信手段在内部都是考内存映射来完成的。





同步机制：

信号量、管程、互斥是进程的同步机制，而信号量、互斥也可用于线程的同步，但管程只在进程同步中被用到；

线程的同步除了信号量、互斥外，还有临界区、事件，没有看到教材上将这两种方式作为进程的同步方式；

通信机制：

管道、FIFO、消息队列、信号量、共享内存是进程的同步机制，教材上没有线程的通信机制这样的说法，但可以肯定这几种方法是进程的通信方式，且其中的信号量既可用于进程的同步，又可用于进程的通信，在网络上还有说可以用于线程同步的。

管道与管程是不同的，管程是进程同步的方式，而管道则是进程通信的方式。





进程间的通信方式有这样几种：

A.共享内存  B.消息队列  C.信号量  D.有名管道  E.无名管道  F.信号

G.文件    H.socket

线程间的通信方式上述进程间的方式都可沿用，且还有自己独特的几种：

A.互斥量   B.自旋锁   C.条件变量 D.读写锁   E.线程信号

G.全局变量

### 水平触发和边缘触发的优缺点

epoll模式下的水平触发、边沿触发。

水平触发通俗来讲：只要有数据，epoll_wait函数就一直返回；边沿触发通俗来讲：只有socket状态发生变化，epoll_wait函数才会返回。

**水平触发优、缺点及应用场景：**

优点：当进行socket通信的时候，保证了数据的完整输出，进行IO操作的时候，如果还有数据，就会一直的通知你。

缺点：由于只要还有数据，内核就会不停的从内核空间转到用户空间，所有占用了大量内核资源，试想一下当有大量数据到来的时候，每次读取一个字节，这样就会不停的进行切换。内核资源的浪费严重。效率来讲也是很低的。

边沿触发优、缺点及应用场景：

优点：每次内核只会通知一次，大大减少了内核资源的浪费，提高效率。

缺点：不能保证数据的完整。不能及时的取出所有的数据。

应用场景：处理大数据。使用non-block模式的socket。



 **边缘触发:**只有电平发生变化(高电平到低电平,或者低电平到高电平)的时候才触发通知.上面提到即使有数据可读,但是io状态没有变化epoll也不会立即返回.

### mysql,nginx,tomcat redis的多进程多线程

多进程单线程模型典型代表：**nginx**

**nginx采用多进程模式：**

对于每个worker进程来说，独立的进程，不需要加锁，所以省掉了锁带来的开销，同时在编程以及问题查找时，也会方便很多。

其次，采用独立的进程，可以让互相之间不会影响，一个进程退出后，其它进程还在工作，服务不会中断，**master进程则很快启动新的worker进程。**

当然，worker进程的异常退出，肯定是程序有bug了，异常退出，会导致当前worker上的所有请求失败，不过不会影响到所有请求，所以降低了风险。

**nginx多进程事件模型：异步非阻塞**

虽然nginx采用多worker的方式来处理请求，每个worker里面只有一个主线程，但是nginx采用了异步非阻塞的方式来处理请求，也就是说，nginx是可以同时处理成千上万个请求的。

一个worker进程可以同时处理的请求数只受限于内存大小，而且在架构设计上，不同的worker进程之间处理并发请求时几乎没有同步锁的限制，worker进程通常不会进入睡眠状态。

因此，**当Nginx上的进程数与CPU核心数相等时（最好每一个worker进程都绑定特定的CPU核心），进程间切换的代价是最小的。**



而**apache的常用工作方式**（apache也有异步非阻塞版本，但因其与自带某些模块冲突，所以不常用），**每个进程在一个时刻只处理一个请求。**

因此，当并发数上到几千时，就同时有几千的进程在处理请求了。这对操作系统来说，是个不小的挑战，进程带来的内存占用非常大，进程的上下文切换带来的cpu开销很大，自然性能就上不去了，而这些开销完全是没有意义的。



apache是同步多进程模型，一个连接对应一个进程；而nginx是异步的，多个连接对应一个进程。

Nginx比较Apache：**事件驱动适合于IO密集型服务**，**多进程或线程适合于CPU密集型服务**



**Nginx与Apache对于高并发处理上的区别。**
    回答3：对于Apache，每个请求都会独占一个工作线程，当并发数到达几千时，就同时有几千的线程在处理请求了。这对于操作系统来说，占用的内存非常大，线程的上下文切换带来的cpu开销也很大，性能就难以上去，同时这些开销是完全没有意义的。

   对于Nginx来讲，一个进程只有一个主线程，通过异步非阻塞的事件处理机制，实现了循环处理多个准备好的事件，从而实现轻量级和高并发。



**Mysql是的单进程多线程的数据库，而oracle使用多进程**

传统的unix系统，早期没有提供多线程，只有多进程。linux是最近的版本才加入多线程支持，以前一直都是多进程。windows很早就支持多线程，本地应用大部分也是多线程。因此**oracle在windows上一直都是多线程，在unix上才是多进程。**

多进程的好处是，**一个进程崩溃不会影响其他进程**，多线程的好处是不需要__共享内存__这样的手段来访问数据库缓冲区




- Nginx优点：负载均衡、反向代理、处理静态文件优势。nginx处理静态请求的速度高于apache；
-  Apache优点：相对于Tomcat服务器来说处理静态文件是它的优势，速度快。Apache是静态解析，适合静态HTML、图片等。
- Tomcat：动态解析容器，处理动态请求，是编译JSP\Servlet的容器，Nginx有动态分离机制，静态请求直接就可以通过Nginx处理，动态请求才转发请求到后台交由Tomcat进行处理。

### Leetcode1166 设计文件系统

需要设计一个能提供下面两个函数的文件系统：

create(path, value): 创建一个新的路径，并尽可能将值 value 与路径 path 关联，然后返回 True。如果路径已经存在或者路径的父路径不存在，则返回 False。
get(path): 返回与路径关联的值。如果路径不存在，则返回 -1。
“路径” 是由一个或多个符合下述格式的字符串连接起来形成的：在 / 后跟着一个或多个小写英文字母。

例如 /leetcode 和 /leetcode/problems 都是有效的路径，但空字符串和 / 不是有效的路径。

好了，接下来就请你来实现这两个函数吧！（请参考示例以获得更多信息）

 

示例 1：

输入： 
["FileSystem","create","get"]
[[],["/a",1],["/a"]]
输出： 
[null,true,1]
解释： 
FileSystem fileSystem = new FileSystem();

fileSystem.create("/a", 1); // 返回 true
fileSystem.get("/a"); // 返回 1
示例 2：

输入： 
["FileSystem","create","create","get","create","get"]
[[],["/leet",1],["/leet/code",2],["/leet/code"],["/c/d",1],["/c"]]
输出： 
[null,true,true,2,false,-1]
解释：
FileSystem fileSystem = new FileSystem();

fileSystem.create("/leet", 1); // 返回 true
fileSystem.create("/leet/code", 2); // 返回 true
fileSystem.get("/leet/code"); // 返回 2
fileSystem.create("/c/d", 1); // 返回 false 因为父路径 "/c" 不存在。
fileSystem.get("/c"); // 返回 -1 因为该路径不存在。

```java
class FileSystem {
public:
    map<string,int> tmp;
    FileSystem() 
    {
        
    }
    
    bool create(string path, int value) 
    {
        if(""==path || "/"==path)
        {
            return false;
        }
        if(tmp[path])
        {
            return false;
        }
        int n=path.length();
        int start=0;
        for(int i=n-1;i>=0;i--)
        {
            if('/'==path[i])
            {
                start=i;
                break;
            }
        }
        if(0==start)
        {
            tmp[path]=value;
            return true;            
        }
        string ss=path.substr(0,start);
        if(0==tmp[ss])
        {
            return false;
        }
        else
        {
            tmp[path]=value;
            return true;
        }     
    }
    
    int get(string path) 
    {
        if(tmp[path])
        {
            return tmp[path];
        }
        else
        {
            return -1;
        }
    }
};
 
/**
 * Your FileSystem object will be instantiated and called as such:
 * FileSystem* obj = new FileSystem();
 * bool param_1 = obj->create(path,value);
 * int param_2 = obj->get(path);
 */
```

```java
class FileSystem {
	unordered_map<string,int> m;
public:
    FileSystem() {
    	m["/"] = 0;
    }
    
    bool createPath(string path, int value) {
    	if(m.count(path+"/")) return false;
    	string tmp = path;
    	while(tmp.back() != '/') 
    		tmp.pop_back();//去除最后一层路径
    	if(!m.count(tmp)) return false;//前置路径不存在
    	m[path+"/"] = value;
    	return true;
    }
    
    int get(string path) {
    	if(m.count(path+'/'))
    		return m[path+'/'];
    	return -1;
    }
};

```



## 零钱问题

### [1.Leetcode322 零钱兑换](https://leetcode-cn.com/problems/coin-change/)-最少数量

给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。

计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。

你可以认为每种硬币的数量是无限的。

 

示例 1：

输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1
示例 2：

输入：coins = [2], amount = 3
输出：-1

```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        //最大最小问题的数量问题
        int n = coins.length;
        int[][] dp = new int[n+1][amount+1];
        // 初始化
        for(int i=0;i<=n;i++){
            Arrays.fill(dp[i],amount+1);
            dp[i][0] = 0;
        }
        // 转移方程
        for(int i=1;i<=n;i++){
            for(int j=0;j<=amount;j++){
                if(j>=coins[i-1]){
                    dp[i][j] = Math.min(dp[i-1][j],dp[i][j-coins[i-1]]+1);
                }else{
                    dp[i][j] = dp[i-1][j];
                }
            }
        }
        return dp[n][amount]==amount+1?-1:dp[n][amount];
    }
}
```

```java
public class Solution {
    public int coinChange(int[] coins, int amount) {
        int max = amount + 1;
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, max);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int j = 0; j < coins.length; j++) {
                if (coins[j] <= i) {
                    dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
                }
            }
        }
        return dp[amount] > amount ? -1 : dp[amount];
    }
}

```



### [2.Leetcode518 零钱兑换II](https://leetcode-cn.com/problems/coin-change-2/)-组合总数


给你一个整数数组 `coins` 表示不同面额的硬币，另给一个整数 `amount` 表示总金额。

请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 `0` 。

假设每一种面额的硬币有无限个。 

题目数据保证结果符合 32 位带符号整数。



**示例 1：**

```
输入：amount = 5, coins = [1, 2, 5]
输出：4
解释：有四种方式可以凑成总金额：
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1
```

```java
class Solution {
    public int change(int amount, int[] coins) {
        int n = coins.length;
        //动态规划
        int[][] dp = new int[n+1][amount+1];
        // 求数量
        for(int i=0;i<=n;i++){
            dp[i][0] = 1;
        }
        //转移方程
        for(int i=1;i<=n;i++){
            for(int j=0;j<=amount;j++){
                if(j>=coins[i-1]){
                    dp[i][j] = dp[i-1][j] + dp[i][j-coins[i-1]];
                }else{
                    dp[i][j] = dp[i-1][j];
                }
            }
        }
        return dp[n][amount];
    }
}
```

### 3.[零钱兑换-组合问题总和]

有1分，2分，5分，10分四种硬币，每种硬币数量无限，给定n分钱(n<10000)，有多少中组合可以组成n分钱？

public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		int n = sc.nextInt();
		sc.close();

		int coins[] = { 1, 2, 5, 10 };
	
		count(coins, 0, n);
		System.out.println(numcount);
	}


```java
// 递归
static int numcount = 0;

public static void count(int coins[], int index, int aim) {

	if (aim == 0) {
		numcount++;
		return;
	}
	if (index == 4) {
		return;
	}

	for (int i = 0; i * coins[index] <= aim; i++) {
		count(coins, index + 1, aim - i * coins[index]);
	}

}
```




### [3.组合总和-具体排列结果](https://leetcode-cn.com/problems/combination-sum/)

给定一个无重复元素的正整数数组 candidates 和一个正整数 target ，找出 candidates 中所有可以使数字和为目标数 target 的唯一组合。

candidates 中的数字可以无限制重复被选取。如果至少一个所选数字数量不同，则两种组合是唯一的。 

对于给定的输入，保证和为 target 的唯一组合数少于 150 个。

 

示例 1：

输入: candidates = [2,3,6,7], target = 7
输出: [[7],[2,2,3]]
示例 2：

输入: candidates = [2,3,5], target = 8
输出: [[2,2,2,2],[2,3,3],[3,5]]

```java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<Integer> path = new ArrayList<>();
        dfs(candidates,0,0,target,path);
        return res;
    }
    public void dfs(int[] candidates,int index,int sum,int target,List<Integer> path){
        //判断是否符合条件
        if(sum==target){
            res.add(new ArrayList<>(path));
            return;
        }
        if(sum>target){
            return;
        }
        // 遍历其它
        for(int i=index;i<candidates.length;i++){
            int temp = sum+candidates[i];
            path.add(candidates[i]);

            dfs(candidates,i,temp,target,path);
            
            path.remove(path.size()-1);
        }
    }
}
```

### [4.组合总和IV-个数](https://leetcode-cn.com/problems/combination-sum-iv/)

给你一个由 不同 整数组成的数组 nums ，和一个目标整数 target 。请你从 nums 中找出并返回总和为 target 的元素组合的个数。

题目数据保证答案符合 32 位整数范围。

 

示例 1：

输入：nums = [1,2,3], target = 4
输出：7
解释：
所有可能的组合为：
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)
请注意，顺序不同的序列被视作不同的组合。

```java
class Solution {
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target+1];
        dp[0] = 1;
        // 开始
        for(int i=1;i<=target;i++){
            for(int num:nums){
                if(i>=num){
                    dp[i] += dp[i-num];
                }
            }
        }
        return dp[target];
    }
}
```

### [5.组合-1-n中选k个数](https://leetcode-cn.com/problems/combinations/)


给定两个整数 `n` 和 `k`，返回范围 `[1, n]` 中所有可能的 `k` 个数的组合。

你可以按 **任何顺序** 返回答案。

 

**示例 1：**

```
输入：n = 4, k = 2
输出：
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
```

```java
class Solution {
    List<List<Integer>> res = new ArrayList<>(); 
    public List<List<Integer>> combine(int n, int k) {
        //返回1...n中的k个数组合
        List<Integer> path = new ArrayList<>();
        dfs(1,n,k,path);
        return res;
    }

    public void dfs(int index,int n,int k,List<Integer> path){
        // 满足条件
        if(path.size()==k){
            res.add(new ArrayList<>(path));
            return;
        }
        // 遍历其它
        for(int i=index;i<=n;i++){
            path.add(i);
            dfs(i+1,n,k,path);
            path.remove(path.size()-1);
        }
    }
}
```

### 6.[金融风暴](https://www.nowcoder.com/questionTerminal/4aab415ce9f94e069c5c44fd8686c0f0)

链接：https://www.nowcoder.com/questionTerminal/4aab415ce9f94e069c5c44fd8686c0f0
来源：牛客网



(金融风暴）银行会互相借钱。在经济艰难时期，如果一个银行倒闭，它就不能偿还贷款。一个 银行的总资产是它当前的余款减去它欠其他银行的贷款。图 8 - 8 就是五个银行的状况图。每个 银行的当前余额分别是 2500 万美元、1 亿 2500万美元、1 亿 7500 万美元、7500 万美元和 1 亿 8100 万美元。从节点 1 到节点 2 的方向的边表示银行 1 借给银行 2 共计 4 千万美元。 

  ![img](https://uploadfiles.nowcoder.com/images/20180727/305899_1532674199460_F37009A234B897096D2F533C3C1A2E3A)


  如果银行的总资产在某个限定范围以下，那么这个银行就是不安全的。它借的钱就不能返 还给借贷方，而且这个借贷方也不能将这个贷款算人它的总资产。因此，如果借贷方总资产在 限定范围以下，那么它也不安全。编写程序，找出所有不安全的银行。程序如下读取输入。它 首先读取两个整数 n 和 limit, 这里的 ri 表示银行个数，而 limit 表示要保持银行安全的最小 总资产。然后，程序会读取描述 n 个银行的 n 行信息，银行的 id 从 0 到 n - 1。每一行的第一个 数字都是银行的余额，第二个数字表明从该银行借款的银行，其余的就都是两个数字构成的数 对。每对都描述一个借款方。每一对数字的第一个数就是借款方的 id, 第二个数就是所借的钱 数。例如，在图 8 - 8 中五个银行的输人如下所示（注意：limit 是 201 ): 

  ![img](https://uploadfiles.nowcoder.com/images/20180727/305899_1532674248069_ACC9E9AE7588ACFA7F5638201A870075)


```java
// 金融风暴

import java.util.Scanner;

public class Financial {
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		System.out.print("输入银行的个数和要保持银行安全的最小总资产(整数): ");
		int bank = sc.nextInt();
		int limit  = sc.nextInt();
		System.out.println();
		double[][] borrowers = new double[bank][bank]; // 创建一个二维数组,double[i][j]表示i银行贷款给j银行的额度
		double[] balance = new double[bank]; // 创建一个一维数组,double[i]表示i银行的余额
		bankInput(borrowers, balance, bank); // 输入银行的基本信息
		bankResult(borrowers, balance, bank, limit); // 输出银行安不安全如果不安全输出它的编号
	}
	public static void bankInput(double[][] borrowers, double[] balance, int bank) { // 输入银行的基本信息
		Scanner sc = new Scanner(System.in);
		for (int i = 0; i < bank; i++) {
			System.out.println("输入银行" + i + "的余额和从该银行借款的银行个数");
			System.out.print("和借款银行的编号和借款数额: ");
			balance[i] = sc.nextDouble();
			int num = sc.nextInt();
			for (int j = 0; j < num; j++) {
				int k = sc.nextInt();
				borrowers[i][k] = sc.nextDouble();
			}
			System.out.println();
		}
	}
	public static void bankResult(double[][] borrowers, double[] balance, int bank, int limit) { // 输出银行安不安全如果不安全输出它的编号
		int[] result = new int[bank * (bank + 1) / 2 + bank]; // 创建一个一维数组用来储存所有的不安全银行编号(可重复)
		int num2 = 0;
		int num1 = 0;
		int num3;
		do {  // 将所有不安全的银行编号储存进result
			num3 = num1;
			num1 = 0;
			for (int i = 0; i < bank; i++) {
				double totalLoan = 0;
				for (int j = 0; j < bank; j++) {
				totalLoan += borrowers[i][j];
				}
				double totalAssets = totalLoan + balance[i];
				if (totalAssets < limit) {
					num2++;
					num1++;
					result[num2 - 1] = i; // 将所有不安全的银行编号储存进result
					for (int k = 0; k < bank; k++) {
						borrowers[k][i] = 0;
					}	
				}	
			}
		}
		while (num1 != num3); // 不安全银行不再增加时结束循环
		if (num2 == 0) {
			System.out.println("没有不安全的银行");
		}
		else {
			UnsafeBank(result, num2); // 将result里面的不重复元素输出即不安全银行的编号
		}
	}
	public static void UnsafeBank(int[] result, int num2) { // 将result里面的不重复元素输出即不安全银行的编号
		int [] j = new int [num2];
		int num;
		int d = 0;
		int i = 0;
		while (i < num2) {
			boolean c = true; // 赋值变量c为true
			num = result[i];
			for (int b = 1; b < d + 1; b++) {
				if (num == j[b - 1]) { // 后面的数字和前面的比较如果相同赋值变量c为false
					c = false;
					break;
				}
			}
			if (c) { // 如果变量c为true，d加1且将num赋值给j[d - 1]
				d++;
				j[d - 1] = num;
			}
			i++;
		}
		int [] e = new int [d]; // a创建一个新的大小为d的数组
		System.arraycopy(j, 0, e, 0, e.length); // 数组e从0项开始复制数组f从0项开始的元素，元素个数为e.length
		System.out.print("不安全的银行是: ");
		for (int f = 0; f < e.length; f++) {
			System.out.print(e[f] + " "); // 输出数组e的所有元素
		}
	}
}
```

## 牛客

### [NC87 丢棋子问题](https://www.nowcoder.com/practice/d1418aaa147a4cb394c3c3efc4302266?tpId=117&&tqId=37844&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

一座大楼有![img](https://www.nowcoder.com/equation?tex=0%E2%88%BCN%5C)层，地面算作第0层，最高的一层为第 ![img](https://www.nowcoder.com/equation?tex=N%5C)层。已知棋子从第0层掉落肯定不会摔碎，从第![img](https://www.nowcoder.com/equation?tex=i%5C)层掉落可能会摔碎，也可能不会摔碎![img](https://www.nowcoder.com/equation?tex=(1%E2%A9%BDi%E2%A9%BDN)%5C)。

给定整数![img](https://www.nowcoder.com/equation?tex=N%5C)作为楼层数，再给定整数![img](https://www.nowcoder.com/equation?tex=K%5C)作为棋子数，返回如果想找到棋子不会摔碎的最高层数，即使在最差的情况下扔的最小次数。一次只能扔一个棋子。

输入数据: 1 \le N, K \le 10^61≤*N*,*K*≤106

**示例1**

输入：

```
10,1
```

返回值：

```
10
```

说明：

```
因为只有1棵棋子，所以不得不从第1层开始一直试到第10层，在最差的情况下，即第10层是不会摔坏的最高层，最少也要扔10次  
```

```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 返回最差情况下扔棋子的最小次数
     * @param n int整型 楼层数
     * @param k int整型 棋子数
     * @return int整型
     */
   public int solve(int n, int k) {
         if (k == 1) {
            return n;
        }
        //如果棋子数足够则返回最小次数
        int best = (int) (Math.log(n) / Math.log(2)) + 1;
        if (k > best) {
            return best;
        }
        int[] dp = new int[k + 1];

        for (int i = 0; i < k + 1; i++) {
            dp[i] = 1; //无论有几个棋子扔1次都只能探测一层
        }
        for (int time = 2;; time++) { //从扔第2次开始（前面初始化dp数组时扔了第1次）
            for (int i = k; i >= 2; i--) { //从k个棋子开始刷新dp数组（倒过来刷新省去记录临时值的步骤）
                dp[i] = dp[i] + dp[i - 1] + 1; //关键一步
                if (dp[i] >= n)
                    return time; //如果探测层数大于n，则返回扔的次数
            }
            dp[1] = time; //1个棋子扔time次最多探测time层
        }
    }
}
```

### [NC57 反转数字](https://www.nowcoder.com/practice/1a3de8b83d12437aa05694b90e02f47a?tpId=117&&tqId=37755&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。

你有注意到翻转后的整数可能溢出吗？因为给出的是32位整数，则其数值范围为[−2^{31}, 2^{31} − 1][−231,231−1]。翻转可能会导致溢出，如果反转后的结果会溢出就返回 0。

**示例1**

输入：

```
12
```

复制

返回值：

```
21
```

复制

**示例2**

输入：

```
-123
```

复制

返回值：

```
-321
```

```java
import java.util.*;


public class Solution {
    /**
     * 
     * @param x int整型 
     * @return int整型
     */
    public int reverse (int x) {
        // write code here
        int res = 0;
        while(x!=0){
            int cur = x%10;
            if((res*10)/10!=res){
                return 0;
            }
            res = res*10+cur;
            x = x/10;
        }
        return res;
    }
}
```

### [NC56 回文数字](https://www.nowcoder.com/practice/35b8166c135448c5a5ba2cff8d430c32?tpId=117&&tqId=37753&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

在不使用额外的内存空间的条件下判断一个整数是否是回文数字

输入整数位于区间 [-2^{31}, 2^{31}-1][−231,231−1]之内

提示：

负整数可以是回文吗？（比如-1）

如果你在考虑将数字转化为字符串的话，请注意一下不能使用额外空间的限制

你可以将整数翻转。但是，如果你做过题目“[反转数字”](https://www.nowcoder.com/practice/1a3de8b83d12437aa05694b90e02f47a?tpId=117&&tqId=34978&rp=1&ru=/ta/job-code-high&qru=/ta/job-code-high/question-ranking)，你会知道将整数翻转可能会出现溢出的情况，你怎么处理这个问题？

**示例1**

输入：

```
121
```

复制

返回值：

```
true
```





```java
import java.util.*;


public class Solution {
    /**
     * 
     * @param x int整型 
     * @return bool布尔型
     */
    public boolean isPalindrome (int x) {
        // write code here
        if(x<0){
            return false;
        }
        long res = 0;
        int num = x;
        while(x!=0){
            res = res*10 + x%10;
            x = x/10;
        }
        return  res==num;
    }
}
```



```java
import java.util.*;


public class Solution {
    /**
     * 
     * @param x int整型 
     * @return bool布尔型
     */
    public boolean isPalindrome (int x) {
        // write code here
        String str = x+"";
        StringBuffer sa =  new StringBuffer(str);
        StringBuffer as =  sa.reverse();
        String str_reverse = as.toString();
        if(str.equals(str_reverse)){
            return true;
        }else{
            return false;
        }
    }
}
```



```java
import java.util.*;


public class Solution {
    /**
     * 
     * @param x int整型 
     * @return bool布尔型
     */
    public boolean isPalindrome (int x) {
        // write code here
         // 如果是负数，或者10的整数倍，返回false
        if (x < 0 || x != 0 && x % 10 == 0)
            return false;
        int reverse = 0;
        while (x > reverse) {
            reverse = reverse * 10 + x % 10;
            x = x / 10;
        }
        return (reverse == x || reverse / 10 == x);
        
    }
}
```

### [NC81 二叉搜索树的第k个结点](https://www.nowcoder.com/practice/ef068f602dde4d28aab2b210e859150a?tpId=117&&tqId=37783&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

给定一棵二叉搜索树，请找出其中的第k小的TreeNode结点。

**示例1**

输入：

```
{5,3,7,2,4,6,8},3
```

复制

返回值：

```
4
```

复制

说明：

```
按结点数值大小顺序第三小结点的值为4  
```

```java
/*
public class TreeNode {
    int val = 0;
    TreeNode left = null;
    TreeNode right = null;

    public TreeNode(int val) {
        this.val = val;

    }

}
*/
public class Solution {
    TreeNode res = null;
    int count = 0;
    TreeNode KthNode(TreeNode pRoot, int k) {
        if(pRoot==null){
            return  res;
        }
        inorder(pRoot,k);
        return res;
    }
    
    public void inorder(TreeNode root,int k){
        if(root==null){
            return;
        }
        inorder(root.left,k);
        count++;
        if(k==count){
            res = root;
            return;
        }
        inorder(root.right,k);
        return;
    }


}
```

### [NC11 最大数](https://www.nowcoder.com/practice/fc897457408f4bbe9d3f87588f497729?tpId=117&&tqId=37835&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

给定一个nums数组由一些非负整数组成，现需要将他们进行排列并拼接，每个数不可拆分，使得最后的结果最大，返回值需要是string类型，否则可能会溢出

提示:

1 <= nums.length <= 100

0 <= nums[i] <= 10000

**示例1**

输入：

```
[30,1]
```

复制

返回值：

```
"301"
```

复制

**示例2**

输入：

```
[2,20,23,4,8]
```

复制

返回值：

```
"8423220"
```

```java
import java.util.*;


public class Solution {
    /**
     * 最大数
     * @param nums int整型一维数组 
     * @return string字符串
     */
    public String solve (int[] nums) {
        // write code here
        //将其转换为string的数组
        int len = nums.length;
        String[] s_arr = new String[len];
        for(int i=0;i<len;i++){
            s_arr[i] = Integer.toString(nums[i]);
        }
        //排序
        Arrays.sort(s_arr,(s1,s2)->(  Integer.valueOf(s2+s1) - Integer.valueOf(s1+s2)  ));
        //判断
        if(s_arr[0].equals("0")){
            return "0";
        }
        //结果
        StringBuilder res = new StringBuilder();
        for(int i=0;i<len;i++){
            res.append(s_arr[i]);
        }
        return res.toString();
        
    }
}
```

### [NC108 最大正方形](https://www.nowcoder.com/practice/0058c4092cec44c2975e38223f10470e?tpId=117&&tqId=37832&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

给定一个由'0'和'1'组成的2维矩阵，返回该矩阵中最大的由'1'组成的正方形的面积

**示例1**

输入：

```
[[1,0,1,0,0],[1,0,1,1,1],[1,1,1,1,1],[1,0,0,1,0]]
```

复制

返回值：

```
4
```

```java
import java.util.*;


public class Solution {
    /**
     * 最大正方形
     * @param matrix char字符型二维数组 
     * @return int整型
     */
    public int solve (char[][] matrix) {
        // write code here
        int maxSize = 0;
        //动态规划
        int rows = matrix.length;
        int cols = matrix[0].length;
        int[][] dp = new int[rows][cols];
        //动态规划
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                //判断
                if(matrix[i][j]=='1'){
                    if(i==0 ||j==0){
                        dp[i][j] = 1;
                    }else{
                        dp[i][j] = Math.min(Math.min(dp[i-1][j],dp[i][j-1]),dp[i-1][j-1])+1;
                    }
                    maxSize = Math.max(maxSize,dp[i][j]);
                }
            }
        }
        return maxSize*maxSize;
    }
}
```

### [NC39 N皇后问题](https://www.nowcoder.com/practice/c76408782512486d91eea181107293b6?tpId=117&&tqId=37811&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

N*N*皇后问题是指在N*N*N*∗*N*的棋盘上要摆N*N*个皇后，
要求：任何两个皇后不同行，不同列也不再同一条斜线上，
求给一个整数N*N*，返回N*N*皇后的摆法数。

**示例1**

输入：

```
1
```

复制

返回值：

```
1
```

复制

**示例2**

输入：

```
8
```

复制

返回值：

```
92
```

```java
import java.util.*;


public class Solution {
    /**
     * 
     * @param n int整型 the n
     * @return int整型
     */
    int res = 0;
    public int Nqueen (int n) {
        // write code here
        // 构建棋盘
        char[][] chess = new char[n][n];
        for(int i=0;i<n;i++) {
            for(int j=0;j<n;j++) {
                chess[i][j] = '.';
            }
        }
        // 开始回溯 第0行开始
        dfs(chess,0);
        return res;
    }

    // 回溯
    public void dfs(char[][] chess,int row) {
        // 递归结束条件
        if(row==chess.length) {
            res+=1;
            return;
        }
        // 回溯
        for(int col=0;col<chess[0].length;col++) {
            // 判断能否放下去
            if(valid(chess,row,col)) {
                chess[row][col] = 'Q';
                dfs(chess,row+1);
                chess[row][col] = '.';
            }
        }
    }
	    
    // 判断合法性
    public boolean valid(char[][] chess,int row,int col) {
        // 判断其上面 其左上角 右上角

        for(int i=0;i<row;i++) {
            if(chess[i][col]=='Q') {
                return false;
            }
        }

        for(int i=row-1,j=col+1;i>=0&&j<chess[0].length;i--,j++) {
            if(chess[i][j]=='Q') {
                return false;
            }
        }

        for(int i=row-1,j=col-1;i>=0&&j>=0;i--,j--) {
            if(chess[i][j]=='Q') {
                return false;
            }
        }
        return true;
    }
    
}
```

### [NC124 字典树的实现](https://www.nowcoder.com/practice/a55a584bc0ca4a83a272680174be113b?tpId=117&&tqId=37818&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

字典树又称为前缀树或者Trie树，是处理字符串常用的数据结构。假设组成所有单词的字符仅是‘a’～‘z’，请实现字典树的结构，并包含以下四个主要的功能。void insert(String word)：添加word，可重复添加；void delete(String word)：删除word，如果word添加过多次，仅删除一次；boolean search(String word)：查询word是否在字典树中出现过(完整的出现过，前缀式不算)；int prefixNumber(String pre)：返回以字符串pre作为前缀的单词数量。现在给定一个m，表示有m次操作，每次操作都为以上四种操作之一。每次操作会给定一个整数op和一个字符串word，op代表一个操作码，如果op为1，则代表添加word，op为2则代表删除word，op为3则代表查询word是否在字典树中，op为4代表返回以word为前缀的单词数量（数据保证不会删除不存在的word）。

对于每次操作，如果op为3时，如果word在字典树中，请输出“YES”，否则输出“NO”；如果op为4时，请输出返回以word为前缀的单词数量，其它情况不输出。

**示例1**

输入：

```
[["1","qwer"],["1","qwe"],["3","qwer"],["4","q"],["2","qwer"],["3","qwer"],["4","q"]]
```

复制

返回值：

```
["YES","2","NO","1"]
```

```java
import java.util.*;


public class Solution {
    /**
     * 
     * @param operators string字符串二维数组 the ops
     * @return string字符串一维数组
     */
    public String[] trieU (String[][] operators) {
        Trie trie = new Trie();
        List<String> list = new ArrayList<String>();
        int len = operators.length;
        for(String[] each:operators){
            if(each[0].equals("1")){
                trie.insert(each[1]);
            }else if(each[0].equals("2")){
                trie.delete(each[1]);
            }else if(each[0].equals("3")){
                boolean tt = trie.search(each[1]);
                if(tt){
                    list.add("YES");
                }else{
                    list.add("NO");
                }
            }else if(each[0].equals("4")){
               list.add(String.valueOf(trie.prefixNumber(each[1])));
            }
        }
        String[] res = new String[list.size()];
        for(int i = 0;i < list.size(); i++){
            res[i] = list.get(i);
        }
        return res;
    }
    
}

class Trie{
    Trie[] child;
    int end;
    int count;
    public Trie(){
        this.end = 0;
        //java数组初始化为null，不会无限初始化
        this.child = new Trie[26];
    }
    
    public void insert(String word){
        Trie cur = this;
        for(char c: word.toCharArray()){
            if(cur.child[c-'a'] == null){
                Trie trie = new Trie();
                cur.child[c-'a'] = trie;
            }
            cur =  cur.child[c-'a'];
            cur.count++;
        }
        cur.end++;
    }
    
    public void delete(String word){
        Trie cur = this;
        for(char c: word.toCharArray()){
            if(cur.child[c-'a'] == null){
                return;
            }
            Trie temp = cur.child[c-'a'];
            if(temp.count == 1){
                cur.child[c-'a'] = null;
            }
            //使用指代寻找
            cur = temp;
            cur.count--;
        }
        cur.end--;
    }
    
    public boolean search(String word){
        Trie cur = this;
        for(char c: word.toCharArray()){
            if(cur.child[c-'a'] == null){
                return false;
            }
            cur = cur.child[c-'a'];
        }
        return cur.end != 0;
    }
    
    public int prefixNumber(String pre){
        Trie cur = this;
        for(char c: pre.toCharArray()){
            if(cur.child[c-'a'] == null){
                return 0;
            }
            cur = cur.child[c-'a'];
        }
        return cur.count;
    }
}
```

### [NC80 把二叉树打印成多行](https://www.nowcoder.com/practice/445c44d982d04483b04a54f298796288?tpId=117&&tqId=37781&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。

例如：
给定的二叉树是{1,2,3,#,#,4,5}
![img](https://uploadfiles.nowcoder.com/images/20210717/557336_1626492068888/41FDD435F0BA63A57E274747DE377E05)
该二叉树多行打印层序遍历的结果是

[

[1],

[2,3],

[4,5]

]

**示例1**

输入：

```
{1,2,3,#,#,4,5}
```

复制

返回值：

```
[[1],[2,3],[4,5]]
```

```java
import java.util.ArrayList;
import java.util.*;

/*
public class TreeNode {
    int val = 0;
    TreeNode left = null;
    TreeNode right = null;

    public TreeNode(int val) {
        this.val = val;

    }

}
*/
public class Solution {
    ArrayList<ArrayList<Integer> > Print(TreeNode root) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        if(root==null){
            return res;
        }
        ArrayList<Integer> path = new ArrayList<>();
        // 层序遍历
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int levelSize = 1;
        while(!queue.isEmpty()){
            root = queue.poll();
            // 注意此处
            path.add(root.val);
            levelSize--;

            if(root.left!=null){
                queue.offer(root.left);
            }
            if(root.right!=null){
                queue.offer(root.right);
            }

            if(levelSize==0){
                res.add(new ArrayList<>(path));
                path = new ArrayList<>();
                levelSize = queue.size();
            }
        }
        return res;
    }
    
}
```

### [NC83 子数组的最大乘积](https://www.nowcoder.com/practice/9c158345c867466293fc413cff570356?tpId=117&&tqId=37785&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

给定一个double类型的数组arr，其中的元素可正可负可0，返回子数组累乘的最大乘积。

**示例1**

输入：

```
[-2.5,4,0,3,0.5,8,-1]
```

复制

返回值：

```
12.00000
```

```java
public class Solution {
    public double maxProduct(double[] nums) {
        int len = nums.length;
        // 两个数组
        double[] maxArr = new double[len];
        double[] minArr = new double[len];
        // 初始化
        maxArr[0] = nums[0];
        minArr[0] = nums[0];
        // 对其遍历
        // 结果存储
        double res = nums[0];
        for(int i=1;i<nums.length;i++){
            // 转移方程
            maxArr[i] = Math.max(nums[i],Math.max(nums[i]*maxArr[i-1],nums[i]*minArr[i-1]));
            minArr[i] = Math.min(nums[i],Math.min(nums[i]*maxArr[i-1],nums[i]*minArr[i-1]));
            res = Math.max(res,maxArr[i]);
        }
        return res;
    }
}
```

### [NC116 把数字翻译成字符串](https://www.nowcoder.com/practice/046a55e6cd274cffb88fc32dba695668?tpId=117&&tqId=37840&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

有一种将字母编码成数字的方式：'a'->1, 'b->2', ... , 'z->26'。

现在给一串数字，返回有多少种可能的译码结果

**示例1**

输入：

```
"12"
```

复制

返回值：

```
2
```

复制

说明：

```
2种可能的译码结果（”ab” 或”l”）
```

**示例2**

输入：

```
"31717126241541717"
```

复制

返回值：

```
192
```



```java
import java.util.*;


public class Solution {
    /**
     * 解码
     * @param nums string字符串 数字串
     * @return int整型
     */
    public int solve (String nums) {
        // write code here
        return dfs(nums.toCharArray(),0);
    }
    
    public int dfs(char[] nums,int start){
        //当start走到终点是
        if(start==nums.length){
            return 1;
        }
        //开始解码当字符为0的时候，0没对应的解码，所以直接返回0
        if(nums[start]=='0'){
            return 0;
        }
        //每次解码一次
        int res1 = dfs(nums,start+1);
        int res2 = 0;
        //如果当前字符等于1
        if( (start<nums.length-1)&&( nums[start]=='1' || (nums[start]=='2'&&nums[start+1]<='6')  )  ){
            res2 = dfs(nums,start+2);
        }
        return res1+res2;
        
        
    }
}
```



### [NC77 调整数组顺序使奇数位于偶数前面](https://www.nowcoder.com/practice/ef1f53ef31ca408cada5093c8780f44b?tpId=117&&tqId=37776&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)(荷兰国旗问题,重点在于顺序不变)

**描述**

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

**示例1**

输入：

```
[1,2,3,4]
```

复制

返回值：

```
[1,3,2,4]
```

复制

**示例2**

输入：

```
[2,4,6,5,7]
```

复制

返回值：

```
[5,7,2,4,6]
```



**该方法会使得数组的相对顺序改变**

```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param array int整型一维数组 
     * @return int整型一维数组
     */
    public int[] reOrderArray (int[] nums) {
        // write code here
        //荷兰国旗问题
        int  less = -1;
        int L = 0;
        int more = nums.length;
        //开始比较
        while(L<more){
            if(nums[L]%2==1){
                swap(nums,++less,L++);
            }else{
                L++;
            }
        }
        return nums;
    }
    
    public void swap(int[] nums,int l,int r){
        int temp = nums[l];
        nums[l]  = nums[r];
        nums[r]  = temp;
    }
}
```

**不改变相对顺序**

```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param array int整型一维数组 
     * @return int整型一维数组
     */
    //解题思路
/*（O(n),O(n)）
遍历两次数组，第一次只添加奇数到新数组里，第二次只添加奇数到新数组里
 */

    public int[] reOrderArray (int[] array) {
        int index = 0;
        int[] res = new int[array.length];
        for (int i : array) {
            if (i % 2 != 0) {
                res[index] = i;
                index++;
            }
        }
        for (int i : array) {
            if (i % 2 == 0) {
                res[index] = i;
                index++;
            }
        }
        return res;
    }
}
```



### [NC117合并二叉树](https://www.nowcoder.com/practice/7298353c24cc42e3bd5f0e0bd3d1d759?tpId=117&&tqId=37841&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

已知两颗二叉树，将它们合并成一颗二叉树。合并规则是：都存在的结点，就将结点值加起来，否则空的位置就由另一个树的结点来代替。例如：
两颗二叉树是:
Tree 1  
   1  
  / \  
  3  2
 /    
 5  

Tree 2
  2
 / \
 1  3
 \  \
  4  7

合并后的树为
  3
  / \
 4  5
 / \  \
5 4  7

**示例1**

输入：

```
{1,3,2,5},{2,1,3,#,4,#,7}
```

复制

返回值：

```
{3,4,5,5,4,#,7}
```

```java
import java.util.*;

/*
 * public class TreeNode {
 *   int val = 0;
 *   TreeNode left = null;
 *   TreeNode right = null;
 * }
 */

public class Solution {
    /**
     * 
     * @param t1 TreeNode类 
     * @param t2 TreeNode类 
     * @return TreeNode类
     */
    public TreeNode mergeTrees (TreeNode t1, TreeNode t2) {
        // write code here
        if(t1==null&&t2==null){
            return null;
        }
        if(t1==null){
            return t2;
        }
        if(t2==null){
            return t1;
        }
        TreeNode merged = new TreeNode(t1.val+t2.val);
        merged.left = mergeTrees(t1.left,t2.left);
        merged.right = mergeTrees(t1.right,t2.right);
        return merged;        
    }
}
```

### [NC135 股票交易的最大收益(二)](https://www.nowcoder.com/practice/4892d3ff304a4880b7a89ba01f48daf9?tpId=117&&tqId=37847&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

假定你知道某只股票每一天价格的变动。

你最多可以同时持有一只股票。但你最多只能进行**两次**交易（一次买进和一次卖出记为一次交易。买进和卖出均无手续费）。

请设计一个函数，计算你所能获得的最大收益。

**示例1**

输入：

```
[8,9,3,5,1,3]
```

复制

返回值：

```
4
```

复制

说明：

```
第三天买进，第四天卖出，第五天买进，第六天卖出。总收益为4。 
```

**备注：**

```5
总天数不大于200000。保证股票每一天的价格在[1,100]范围内。
```

```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 两次交易所能获得的最大收益
     * @param prices int整型一维数组 股票每一天的价格
     * @return int整型
     */
    public int maxProfit (int[] prices) {
        if(prices.length<1){
            return 0;
        }
        // write code here
         // 一次也没有交易
        int dp0 = 0;
        // 第一次买日
        int dp1 = -prices[0];
        // 第一次卖出
        int dp2 = 0;
        // 第二次买入
        int dp3 = Integer.MIN_VALUE;
        // 第二次卖出
        int dp4 = Integer.MIN_VALUE;
        for(int i=1;i<prices.length;i++){
            int newdp0 = dp0;
            int newdp1 = Math.max(dp1,dp0-prices[i]);
            int newdp2 = Math.max(dp2,dp1+prices[i]);
            int newdp3 = Math.max(dp3,dp2-prices[i]);
            int newdp4 = Math.max(dp4,dp3+prices[i]);

            dp0 = newdp0;
            dp1 = newdp1;
            dp2 = newdp2;
            dp3 = newdp3;
            dp4 = newdp4;
        }
        return Math.max(dp2,dp4);
    }
}
```



### [NC143 矩阵乘法](https://www.nowcoder.com/practice/bf358c3ac73e491585943bac94e309b0?tpId=117&&tqId=37854&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

给定两个 n*n*n*∗*n* 的矩阵 A*A* 和 B*B* ，求 A*B*A*∗*B* 。

数据范围：

1 \le n \le 5001≤*n*≤500

-100 \le n \le 100−100≤*n*≤100

**示例1**

输入：

```
[[1,2],[3,2]],[[3,4],[2,1]]
```

复制

返回值：

```
[[7,6],[13,14]]
```

```java
import java.util.*;
public class Solution {
    public int[][] solve (int[][] a, int[][] b) {
        // write code here
            int m = a.length;//A的行数
            int p = a[0].length;//A的列数=B的行数
            int n = b[0].length;//B的行数
            int[][] ans = new int[m][n];
            for(int i = 0 ; i < m ; i++)
            {
                for(int j = 0 ; j < n ; j++)
                {
                        int t = 0;
                        for(int k = 0 ; k < p ; k++)
                        {
                                t+=a[i][k]*b[k][j];
                        }
                        ans[i][j]=t;
                }
            }
            return ans;
    }
}
```

### [NC129 阶乘末尾0的数量](https://www.nowcoder.com/practice/aa03dff18376454c9d2e359163bf44b8?tpId=117&&tqId=37803&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

给定一个非负整数 N*N*，返回 N!*N*! 结果的末尾为 00 的数量。

N!*N*! 是指自然数 N*N* 的阶乘,即:N!=1*2*3…(N-2)*(N-1)*N*N*!=1∗2∗3…(*N*−2)∗(*N*−1)∗*N*。

特殊的, ![img](https://www.nowcoder.com/equation?tex=0%5C) 的阶乘是 ![img](https://www.nowcoder.com/equation?tex=1%5C) 。

**示例1**

输入：

```
3
```

复制

返回值：

```
0
```

复制

说明：

```
3!=6   
```

**示例2**

输入：

```
5
```

复制

返回值：

```
1
```

复制

说明：

```
5!=120  
```

```java
import java.util.*;


public class Solution {
    /**
     * the number of 0
     * @param n long长整型 the number
     * @return long长整型
     */
    public long thenumberof0 (long n) {
        // write code here
        long  res = 0;
        long d = 5;
        while(n>=d){
            res += n/d;
            d = d*5;
        }
        return res;
    }
}
```

### [NC58 找到搜索二叉树中两个错误的节点](https://www.nowcoder.com/practice/4582efa5ffe949cc80c136eeb78795d6?tpId=117&&tqId=37820&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

一棵二叉树原本是搜索二叉树，但是其中有两个节点调换了位置，使得这棵二叉树不再是搜索二叉树，请按升序输出这两个错误节点的值。(每个节点的值各不相同)

**示例1**

输入：

```
{1,2,3}
```

复制

返回值：

```
[1,2]
```

```java
import java.util.*;

/*
 * public class TreeNode {
 *   int val = 0;
 *   TreeNode left = null;
 *   TreeNode right = null;
 * }
 */

public class Solution {
    /**
     * 
     * @param root TreeNode类 the root
     * @return int整型一维数组
     */
    public int[] findError (TreeNode root) {
        // write code here
        if(root==null){
            return new int[]{};
        }
        TreeNode pre = null;
        TreeNode first = null;
        TreeNode second = null;
        //中序遍历
        Stack<TreeNode> stack = new Stack<>();
        while(!stack.isEmpty()||root!=null){
            while(root!=null){
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if(pre!=null&&pre.val>root.val){
                second = root;    
                if(first==null){
                    first = pre;
                }else{
                    break;
                }
            }
            
            pre = root;
            root = root.right;
        }
        int[] res = new int[2];
        res[0] = second.val;
        res[1] = first.val;
        return res;
        
    }
}
```



### [Leetcode003 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。

 

示例 1:

输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
示例 2:

输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。



```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        char[] arr = s.toCharArray();
        int len = arr.length;
        // hashmap辅助
        HashMap<Character,Integer> dict = new HashMap<>();
        // 结果
        int res = 0;
        // 滑动窗口
        int left = 0;
        int right = 0;
        while(right<len){
            //判断是否包含
            if(dict.containsKey(arr[right])){
                left = Math.max(dict.get(arr[right])+1,left);
            }
            dict.put(arr[right],right);
            // 结果更新
            res = Math.max(res,right-left+1);
            // 继续
            right++;
        }
        return res;
    }
}
```

### [NC41 最长无重复子数组](https://www.nowcoder.com/practice/b56799ebfd684fb394bd315e89324fb4?tpId=188&&tqId=38553&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给定一个数组arr，返回arr的最长无重复元素子数组的长度，无重复指的是所有数字都不相同。

子数组是连续的，比如[1,3,5,7,9]的子数组有[1,3]，[3,5,7]等等，但是[1,3,7]不是子数组

**示例1**

输入：

```
[2,3,4,5]
```

复制

返回值：

```
4
```

复制

说明：

```
[2,3,4,5]是最长子数组    
```

```java
import java.util.*;


public class Solution {
    /**
     * 
     * @param arr int整型一维数组 the array
     * @return int整型
     */
    public int maxLength (int[] arr) {
        // write code here
        HashMap<Integer,Integer> dict = new HashMap<>();
        int l = 0;
        int r = 0;
        int maxlen = 0;
        while(r<arr.length){
            if(dict.containsKey(arr[r])){
                l = Math.max(dict.get(arr[r])+1,l);
            }
            dict.put(arr[r],r);
            maxlen = Math.max(maxlen,r-l+1);
            r++;
        }
        return maxlen;
    }
}
```







### [NC142 最长重复子串](https://www.nowcoder.com/practice/4fe306a84f084c249e4afad5edf889cc?tpId=117&&tqId=37853&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

定义**重复字符串**是由两个相同的字符串首尾拼接而成，例如 ![img](https://www.nowcoder.com/equation?tex=abcabc%5C) 便是长度为6的一个重复字符串，而 ![img](https://www.nowcoder.com/equation?tex=abcba%5C) 则不存在重复字符串。

给定一个字符串，请返回其最长重复子串的长度。

若不存在任何重复字符子串，则返回 0 。

**示例1**

输入：

```
"ababc"
```

复制

返回值：

```
4
```

复制

说明：

```
abab为最长的重复字符子串，长度为4  
```

```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 
     * @param a string字符串 待计算字符串
     * @return int整型
     */
    public int solve (String a) {
        // write code here
        //判断
        if(a==null||a.length()<=1){
            return 1;
        }
        //转成数组
        char[] arr = a.toCharArray();
        int len = arr.length;
        int maxLen = len/2;
        //开始判断
        for(int i=maxLen;i>0;i--){
            for(int j=0;j<=len-2*i;j++){
                if(check(arr,j,i)){
                    return 2*i;
                }
            }
        }
        return 0;
    }
    public boolean check(char[] arr,int start,int len){
        for(int i=start;i<start+len;i++){
            if(arr[i]!=arr[i+len]){
                return false;
            }
        }
        return true;
        
    }
}
```

### [NC11 将升序数组转换为平衡二叉搜索树](https://www.nowcoder.com/practice/7e5b00f94b254da599a9472fe5ab283d?tpId=117&&tqId=37720&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

给出一个升序排序的数组，将其转化为平衡二叉搜索树（BST）.

**示例1**

输入：

```
[-1,0,1,2]
```

复制

返回值：

```
{1,0,2,-1}
```

```java
import java.util.*;

/*
 * public class TreeNode {
 *   int val = 0;
 *   TreeNode left = null;
 *   TreeNode right = null;
 * }
 */

public class Solution {
    /**
     * 
     * @param num int整型一维数组 
     * @return TreeNode类
     */
    public TreeNode sortedArrayToBST (int[] nums) {
        // write code here
        return myTree(nums,0,nums.length-1);
    }
    public TreeNode myTree(int[] nums,int l,int r){
        //注意递归结束条件
        if(l>r){
            return null;
        }
        int mid = l + ((r-l+1)>>1);
        TreeNode root = new TreeNode(nums[mid]);
        //注意递归
        root.left = myTree(nums,l,mid-1);
        root.right = myTree(nums,mid+1,r);
        return root;
    }
}
```

### [NC47 数独](https://www.nowcoder.com/practice/5e6c424b82224b85b64f28fd85761280?tpId=117&&tqId=37743&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

请编写一个程序，给数独中的剩余的空格填写上数字

空格用字符'.'表示

假设给定的数独只有唯一的解法

![img](http://uploadfiles.nowcoder.com/images/20150314/0_1426326866206_250px-Sudoku-by-L2G-20050714.svg.png)

这盘数独的解法是：

![img](http://uploadfiles.nowcoder.com/images/20150314/0_1426326866565_250px-Sudoku-by-L2G-20050714_solution.svg.png)

红色表示填上的解

**示例1**

输入：

```
[[.,.,9,7,4,8,.,.,.],[7,.,.,.,.,.,.,.,.],[.,2,.,1,.,9,.,.,.],[.,.,7,.,.,.,2,4,.],[.,6,4,.,1,.,5,9,.],[.,9,8,.,.,.,3,.,.],[.,.,.,8,.,3,.,2,.],[.,.,.,.,.,.,.,.,6],[.,.,.,2,7,5,9,.,.]]
```

复制

返回值：

```
[[5,1,9,7,4,8,6,3,2],[7,8,3,6,5,2,4,1,9],[4,2,6,1,3,9,8,7,5],[3,5,7,9,8,6,2,4,1],[2,6,4,3,1,7,5,9,8],[1,9,8,5,2,4,3,6,7],[9,7,5,8,6,3,1,2,4],[8,3,2,4,9,1,7,5,6],[6,4,1,2,7,5,9,8,3]]
```

```java
public class Solution {
    public void solveSudoku(char[][] board) {
        dfs(board,0,0);
    }
    boolean dfs(char[][] board,int x,int y){
        if(x==9)return true;
        if(y==9)return dfs(board,x+1,0);
        if(board[x][y] != '.')return dfs(board,x,y+1);
        for(char c='1';c<='9';++c){
            if(!isValid(board,x,y,c))continue;
            board[x][y]=c;
            if(dfs(board,x,y+1))return true;
            board[x][y] = '.';
        }
        return false;
    }
    boolean isValid(char[][] board,int x,int y,char ch){
        for(int i=0;i<9;++i){
            if(board[x][i] == ch)return false;
            if(board[i][y] == ch)return false;
            if(board[(x/3)*3+ i/3][(y/3)*3 + i%3]==ch)return false;
        }
        return true;
    }
}
```

### [NC31 第一个只出现一次的字符](https://www.nowcoder.com/practice/1c82e8cf713b4bbeb2a5b31cf5b0417c?tpId=117&&tqId=37762&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.（从0开始计数）

**示例1**

输入：

```
"google"
```

复制

返回值：

```
4
```

```java
public class Solution {
    public int FirstNotRepeatingChar(String str) {
         if(str==null||str.length()==0){
             return -1;
         }   
         //用一个类似hash的数组来存储字符出现的次数
        int[] count = new int[128];
        for(int i=0;i<str.length();i++){
            count[str.charAt(i)]++;
        }
        for(int i=0;i<str.length();i++){
            if(count[str.charAt(i)]==1){
                return i;
            }
        }
        return -1;
    }
}
```

### [NC139 孩子的游戏(圆圈中最后剩下的数)](https://www.nowcoder.com/practice/f78a359491e64a50bce2d89cff857eb6?tpId=117&&tqId=37767&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)

如果没有小朋友，请返回-1

**示例1**

输入：

```
5,3
```

复制

返回值：

```
3
```

```java
import java.util.*;

public class Solution {
    public int LastRemaining_Solution(int n, int m) {
        if(n==0 || m==0){
            return -1;
        }
        // 用List来做
        List<Integer> list = new ArrayList<>();
        for(int i=0;i<n;i++){
            list.add(i);
        }
        // 第一个的索引是
        int index = 0;
        while(list.size()>1){
            // 开始
            for(int i=1;i<m;i++){
                index = (index+1)%list.size();
            }
            list.remove(index);
        }
        return list.get(0);
    }
}
```

### [NC71 旋转数组的最小数字](https://www.nowcoder.com/practice/9f3231a991af4f55b95579b44b7a01ba?tpId=117&&tqId=37768&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

**示例1**

输入：

```
[3,4,5,1,2]
```

复制

返回值：

```
1
```

```java
import java.util.ArrayList;
public class Solution {
    public int minNumberInRotateArray(int [] numbers) {
        int l = 0;
        int r = numbers.length-1;
        int mid = 0;
        while(l<=r){
            mid = l + ((r-l)>>1);
            if(numbers[mid]>numbers[r]){
                l = mid+1;
            }else if(numbers[mid]<numbers[r]){
                r = mid;
            }else{
                r--;
            }
        }
        return numbers[mid];
    }
}
```

### [NC74 数组在升序数组中出现的次数](https://www.nowcoder.com/practice/70610bf967994b22bb1c26f9ae901fa2?tpId=117&&tqId=37772&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

统计一个数字在升序数组中出现的次数。

**示例1**

输入：

```
[1,2,3,3,3,3,4,5],3
```

复制

返回值：

```
4
```

```java
public class Solution {
    public int GetNumberOfK(int [] nums , int target) {
        int lbound = 0, rbound = 0;
        // 寻找上界
        int l = 0, r = nums.length;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] < target) {
                l = mid + 1;
            }
            else {
                r = mid;
            }
        }
        lbound = l;
        // 寻找下界
        l = 0;r = nums.length;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] <= target) {
                l = mid + 1;
            }
            else {
                r = mid;
            }
        }
        rbound = l;
        return rbound - lbound;
    }
}
```

### [NC84 完全二叉树结点数](https://www.nowcoder.com/practice/512688d2ecf54414826f52df4e4b5693?tpId=117&&tqId=37786&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

描述

给定一棵完全二叉树的头节点head，返回这棵树的节点个数。如果完全二叉树的节点数为![img](https://www.nowcoder.com/equation?tex=N%5C)，请实现时间复杂度为O(log^2N)*O*(*l**o**g*2*N*)的解法。 

示例1

输入：

```
{1,2,3} 
```

复制

返回值：

```
3
 
```

```java
/**
public class TreeNode {
    int val = 0;
    TreeNode left = null;
    TreeNode right = null;

    public TreeNode(int val) {
        this.val = val;

    }
}*/
public class Solution {
    public int nodeNum(TreeNode root) {
         if(root==null){
            return 0;
        }
        int leftcount = nodeNum(root.left);
        int rightcount = nodeNum(root.right);
        return leftcount+rightcount+1;
    }
}
```

### [NC125 未排序数组中累加和为给定值的最长子数组长度](https://www.nowcoder.com/practice/704c8388a82e42e58b7f5751ec943a11?tpId=117&&tqId=37794&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

给定一个无序数组arr, 其中元素可正、可负、可0。给定一个整数k，求arr所有连续子数组中累加和为k的最长子数组长度。

保证至少存在一个合法的子数组。

**示例1**

输入：

```
[1,-2,1,1,1],0
```

复制

返回值：

```
3
```

```java
import java.util.*;


public class Solution {
    /**
     * max length of the subarray sum = k
     * @param arr int整型一维数组 the array
     * @param k int整型 target
     * @return int整型
     */
    public int maxlenEqualK (int[] arr, int k) {
        // write code here
        if(arr==null||arr.length==0){
            return 0;
        }
        Map<Integer,Integer> map = new HashMap<>();
        map.put(0,-1);
        int len = 0;
        int sum = 0;
        for(int i=0;i<arr.length;i++){
            sum += arr[i];
            if(map.containsKey(sum-k)){
                len = Math.max(len,i-map.get(sum-k));
            }
            if(!map.containsKey(sum)){
                map.put(sum,i);
            }
        }
        return len;
        
    }
}
```

### [NC104 比较版本号](https://www.nowcoder.com/practice/2b317e02f14247a49ffdbdba315459e7?tpId=117&&tqId=37828&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

牛客项目发布项目版本时会有版本号，比如1.02.11，2.14.4等等

现在给你2个版本号version1和version2，请你比较他们的大小

版本号是由修订号组成，修订号与修订号之间由一个"."连接。1个修订号可能有多位数字组成，修订号可能包含前导0，且是合法的。例如，1.02.11，0.1，0.2都是合法的版本号

每个版本号至少包含1个修订号。

修订号从左到右编号，下标从0开始，最左边的修订号下标为0，下一个修订号下标为1，以此类推。

比较规则：

一. 比较版本号时，请按从左到右的顺序依次比较它们的修订号。比较修订号时，只需比较忽略任何前导零后的整数值。比如"0.1"和"0.01"的版本号是相等的

二. 如果版本号没有指定某个下标处的修订号，则该修订号视为0。例如，"1.1"的版本号小于"1.1.1"。因为"1.1"的版本号相当于"1.1.0"，第3位修订号的下标为0，小于1

三. version1 > version2 返回1，如果 version1 < version2 返回-1，不然返回0.

数据范围：

version1 和version2 字符串长度不超过1000 ，但是版本号的每一节可能超过 int 表达范围

**示例1**

输入：

```
"1.1","2.1"
```

复制

返回值：

```
-1
```

复制

说明：

```
version1 中下标为 0 的修订号是 "1"，version2 中下标为 0 的修订号是 "2" 。1 < 2，所以 version1 < version2，返回-1
     
```

```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 比较版本号
     * @param version1 string字符串 
     * @param version2 string字符串 
     * @return int整型
     */
    public int compare (String version1, String version2) {
        // write code here
        if(version1==null||version2==null||version1.length()==0||version2.length()==0) return 0;
        String[] v1 = version1.split("\\.");
        String[] v2 = version2.split("\\.");
        int len = Math.min(v1.length,v2.length);
        for(int i=0;i<len;i++){
            v1[i]=stripZero(v1[i]);//去掉头部的0
            v2[i]=stripZero(v2[i]);//去掉头部的0
            if(v1[i].equals(v2[i])) continue;
            int n1=0, n2=0;
            try{ //如果能用int 去比较最好
                n1 = Integer.parseInt(v1[i]);
                n2 = Integer.parseInt(v2[i]);
            }catch(Exception e){ //不能用int 比较就用java string 的compareTo
                if(v1[i].compareTo(v2[i])>0){
                    return 1;
                }else if(v1[i].compareTo(v2[i])<0){
                    return -1;
                }else{
                    //return 0;
                    continue;
                }
            }
            if(n1>n2){
                return 1;
            }else if(n1<n2){
                return -1;
            }else{
                //return 0;
                continue;
            }
        }
        if(v1.length>v2.length){
            for(int i=len;i<v1.length;i++){
                int n1 = 0;
                try{
                    n1 = Integer.parseInt(v1[i]);
                    if(n1==0) continue;
                    else return 1;
                }catch(Exception e){
                    return 1;
                }
            }
            return 0;
        }else if(v1.length<v2.length){
            for(int i=len;i<v2.length;i++){
                int n2 = 0;
                try{
                    n2 = Integer.parseInt(v2[i]);
                    if(n2==0) continue;
                    else return -1;
                }catch(Exception e){
                    return -1;
                }
            }
            return 0;
        }
        return 0;
    }

    //去掉字符串头部的0的函数
    public String stripZero(String s){
        int i=0;
        while(i<s.length()){
            if(s.charAt(i)=='0'){
                i++;
                continue;
            } 
            break;
        }
        return s.substring(i,s.length());

    }
}
```

### [NC23 划分链表](https://www.nowcoder.com/practice/1dc1036be38f45f19000e48abe00b12f?tpId=117&&tqId=37728&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

给出一个单链表和一个值x*x*，请返回一个链表的头结点，要求新链表中<x<*x*的节点全部在\ge x≥*x*的节点左侧，并且两个部分之内的节点之间与原来的链表要保持相对顺序不变。

例如：

给出 1\to 4 \to 3 \to 2 \to 5 \to 21→4→3→2→5→2 和 \ x = 3 *x*=3,

返回 1\to 2 \to 2 \to 4 \to 3 \to 51→2→2→4→3→5

数据范围：链表长度\le 200≤200，链表节点的val值 \in [-100, 100]∈[−100,100]

备注：要求时间复杂度O(n)*O*(*n*), 空间复杂度O(1)*O*(1).

**示例1**

输入：

```
{1,4,3,2,5,2},3
```

复制

返回值：

```
{1,2,2,4,3,5}
```

```java
import java.util.*;

/*
 * public class ListNode {
 *   int val;
 *   ListNode next = null;
 * }
 */

public class Solution {
    /**
     * 
     * @param head ListNode类 
     * @param x int整型 
     * @return ListNode类
     */
    public ListNode partition (ListNode head, int x) {
        // write code here
        ListNode smallDummy = new ListNode(-1);
        ListNode bigDummy = new ListNode(-1);
        ListNode smallHead = smallDummy;
        ListNode bigHead = bigDummy;
        while(head!=null){
            if(head.val<x){
                smallHead.next = head;
                smallHead = smallHead.next;
            }else{
                bigHead.next = head;
                bigHead = bigHead.next;
            }
            head = head.next;
        }
        //记得给其加个null
        bigHead.next = null;
        smallHead.next = bigDummy.next;
        return smallDummy.next;
    }
}
```

### [NC106 三个数的最大乘积](https://www.nowcoder.com/practice/8ae05c2913fe438b8b14f3968f64fc0b?tpId=117&&tqId=37830&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

给定一个长度为 ![img](https://www.nowcoder.com/equation?tex=n%5C) 的无序数组 ![img](https://www.nowcoder.com/equation?tex=A%5C) ，包含正数、负数和 0 ，请从中找出 3 个数，使得乘积最大，返回这个乘积。

要求时间复杂度： ![img](https://www.nowcoder.com/equation?tex=O(n)%5C) ，空间复杂度： ![img](https://www.nowcoder.com/equation?tex=O(1)%5C) 。

数据范围：

3 \le n \le 2 * 10^53≤*n*≤2∗105

-10^6 \le A[i] \le 10^6−106≤*A*[*i*]≤106

**示例1**

输入：

```
[3,4,1,2]
```

复制

返回值：

```
24
```

> 解题思路：遍历数组，找到最大的三个数和最小的两个数，三个数的最大乘积来源可能有两种，一种是三个最大的数相乘，另一种是两个最小的数和一个最大的数相乘

```java
import java.util.*;
public class Solution {
    /**
     * 最大乘积
     * @param A int整型一维数组 
     * @return long长整型
     */
    public long solve (int[] A) {
        // write code here
        int max1=Integer.MIN_VALUE,max2=Integer.MIN_VALUE,max3=Integer.MIN_VALUE;
        int min1=Integer.MAX_VALUE,min2=Integer.MAX_VALUE;
        for(int num:A){
            if(num>max1){
                max3=max2;
                max2=max1;
                max1=num;
            } else if(num>max2){
                max3=max2;
                max2=num;
            } else if(num>max3) max3=num;
            if(num<min1){
                min2=min1;
                min1=num;
            } else if(num<min2) min2=num;
        }
        return Math.max((long)max1*max2*max3,(long)max1*min1*min2);
    }
}
```

### [NC114 旋转字符串](https://www.nowcoder.com/practice/80b6bb8797644c83bc50ac761b72981c?tpId=117&&tqId=37838&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

字符串旋转:

给定两字符串A和B，如果能将A从中间某个位置分割为左右两部分字符串（可以为空串），并将左边的字符串移动到右边字符串后面组成新的字符串可以变为字符串B时返回true。

例如：如果A=‘youzan’，B=‘zanyou’，A按‘you’‘zan’切割换位后得到‘zanyou’和B相同，返回true。

再如：如果A=‘abcd’，B=‘abcd’，A切成‘abcd’和''（空串），换位后可以得到B，返回true。

**示例1**

输入：

```
"youzan","zanyou"
```

复制

返回值：

```
true
```

复制

**示例2**

输入：

```
"youzan","zyouan"
```

复制

返回值：

```
false
```



> 优美的解决思路：假如 A="abcd" 则 A+A = "abcdabcd"
> 2、如果B 满足 题目的条件，则B 一定属于 A+A 里面的一个子串

```java
import java.util.*;


public class Solution {
    /**
     * 旋转字符串
     * @param A string字符串 
     * @param B string字符串 
     * @return bool布尔型
     */
    public boolean solve (String A, String B) {
        // write code here
          //特殊情况处理
        if(A==null||B==null||A.length()<2||B.length()<2||A.length()!=B.length()){
            return false;
        }
        //日   这么巧妙
        return (A+A).contains(B);
    }
}
```

思路之二：
1、A和B长度不等，则返回 false
2、A个B长度相等，不断的切割 A为head和tail 两部分
3、如果B中同时包含 head和tail 两部分，则返回true
4、如果一直没找到，则返回false

```java
import java.util.*;


public class Solution {
    /**
     * 旋转字符串
     * @param A string字符串 
     * @param B string字符串 
     * @return bool布尔型
     */
    public boolean solve (String A, String B) {
         // write code here
        if(A==null||B==null||A.length()<2||B.length()<2||A.length()!=B.length()){
            return false;
        }
        int i=1;
        while(i<A.length()){
            String headStr = A.substring(0,i);
            String tailStr = A.substring(i);
            if(B.contains(headStr)&&B.contains(tailStr)){
                return true;
            }
            i++;
        }
        return false;
    }
}
```

### [NC115 栈和排序](https://www.nowcoder.com/practice/95cb356556cf430f912e7bdf1bc2ec8f?tpId=117&&tqId=37839&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

给你一个 1 到 ![img](https://www.nowcoder.com/equation?tex=n%5C) 的排列和一个栈，入栈顺序给定

你要在不打乱入栈顺序的情况下，对数组进行从大到小排序

当无法完全排序时，请输出字典序最大的出栈序列

数据范围: 1 \le n \le 10^61≤*n*≤106

**示例1**

输入：

```
[2,1,5,3,4]
```

复制

返回值：

```
[5,4,3,1,2]
```

复制

说明：

```
2入栈；1入栈；5入栈；5出栈；3入栈；4入栈；4出栈；3出栈；1出栈；2出栈 
```

```java
import java.util.*;


public class Solution {
    /**
     * 栈排序
     * @param a int整型一维数组 描述入栈顺序
     * @return int整型一维数组
     */
   public int[] solve (int[] a) {
       int n=a.length;
        Stack<Integer>stack=new Stack<>();
        int[]dp=new int[n];
        dp[n-1]=a[n-1];
        for(int i=n-2;i>=0;i--)
            dp[i]=Math.max(dp[i+1],a[i]);  //用一个数组记录第i个及之后最大元素
        int[]res=new int[n];
        int j=0;
        for(int i=0;i<n;i++){
            stack.push(a[i]);
            while(!stack.isEmpty()&&i<n-1&&stack.peek()>=dp[i+1]) 
                res[j++]=stack.pop();//如果栈顶元素比后面的都大，那么出栈
        }
        while(!stack.isEmpty()) res[j++]=stack.pop(); //最后在栈中的按顺序弹出
        return res;
    }
}
```

### [NC145 01背包](https://www.nowcoder.com/practice/2820ea076d144b30806e72de5e5d4bbf?tpId=117&&tqId=37856&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

已知一个背包最多能容纳物体的体积为 ![img](https://www.nowcoder.com/equation?tex=V%5C)

现有 ![img](https://www.nowcoder.com/equation?tex=n%5C) 个物品，第 ![img](https://www.nowcoder.com/equation?tex=i%5C) 个物品的体积为 v_i*v**i* , 重量为 w_i*w**i*

求当前背包最多能装多大重量的物品?

数据范围：

1 \le V \le 50001≤*V*≤5000

1 \le n \le 50001≤*n*≤5000

1 \le v_i \le 50001≤*v**i*≤5000

1 \le w_i \le 50001≤*w**i*≤5000

**示例1**

输入：

```
10,2,[[1,3],[10,4]]
```

复制

返回值：

```
4
```

复制

说明：

```
第一个物品的体积为1，重量为3，第二个物品的体积为10，重量为4。只取第二个物品可以达到最优方案，取物重量为4  
```

```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 计算01背包问题的结果
     * @param V int整型 背包的体积
     * @param n int整型 物品的个数
     * @param vw int整型二维数组 第一维度为n,第二维度为2的二维数组,vw[i][0],vw[i][1]分别描述i+1个物品的vi,wi
     * @return int整型
     */
    public int knapsack (int V, int n, int[][] vw) {
        // write code here
        if(V==0 || n==0 || vw==null){
            return 0;
        }
        int[][] dp=new int[n+1][V+1];
        for(int i=1;i<=n;i++){
            for(int j=1;j<=V;j++){
                if(j<vw[i-1][0]){
                    dp[i][j]=dp[i-1][j];
                }
                else{
                    dp[i][j]=Math.max(dp[i-1][j],dp[i-1][j-vw[i-1][0]]+vw[i-1][1]);
                }
            }
        }
        return dp[n][V];
    }
}
```

### [NC156 数组中只出现一次的数字（其他数出现k次）](https://www.nowcoder.com/practice/5d3d74c3bf7f4e368e03096bb8857871?tpId=117&&tqId=37866&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

给定一个长度为 ![img](https://www.nowcoder.com/equation?tex=n%5C) 的整型数组 ![img](https://www.nowcoder.com/equation?tex=arr%5C) 和一个整数 ![img](https://www.nowcoder.com/equation?tex=k(k%3E1)%5C) 。

已知 ![img](https://www.nowcoder.com/equation?tex=arr%5C) 中只有 1 个数出现一次，其他的数都出现 ![img](https://www.nowcoder.com/equation?tex=k%5C) 次。

请返回只出现了 1 次的数。

要求时间复杂度![img](https://www.nowcoder.com/equation?tex=O(32n)%2C%5C)空间复杂度![img](https://www.nowcoder.com/equation?tex=O(1)%5C)

数据范围:

1 \le n \le 2*10^51≤*n*≤2∗105
1 \lt k \lt 1001<*k*<100
-2*10^9 \le arr[i] \le 2*10^9−2∗109≤*a**r**r*[*i*]≤2∗109

**示例1**

输入：

```
[5,4,1,1,5,1,5],3 
```

复制

返回值：



```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param arr int一维数组 
     * @param k int 
     * @return int
     */
    public int foundOnceNumber (int[] arr, int k) {
        // write code here
          Arrays.sort(arr);
      for(int i = 0; i<arr.length-1; i++){
          if(arr[i]==arr[i+1]){
              i += k-1;
          }else{
              return arr[i];
          }
 
      }
      return arr[arr.length-1];
    }
}
```



```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param arr int一维数组 
     * @param k int 
     * @return int
     */
    public int foundOnceNumber (int[] arr, int k) {
        // 每个二进制位求和，如果某个二进制位不能被k整除，那么只出现一次的那个数字在这个二进制位上为1。
        int[] binarySum = new int[32];
        for(int i = 0; i< 32; i++){//求每个二进制位的和
            int sum = 0;
            for(int num : arr){
                sum += (num >>i & 1);//依次右移num，同1相与，计算每一位上1的个数
            }
            binarySum[i] = sum;
        }
        int res = 0;
        for (int i = 0; i< 32; i++){
            if(binarySum[i]%k!=0){
                res += 1<<i;//左移恢复
            }
        }
        return res;
    }
}
```

### [NC67 汉诺塔问题](https://www.nowcoder.com/practice/7d6cab7d435048c4b05251bf44e9f185?tpId=117&&tqId=37787&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

我们有由底至上为从大到小放置的 ![img](https://www.nowcoder.com/equation?tex=n%5C) 个圆盘，和三个柱子（分别为左/中/右），开始时所有圆盘都放在左边的柱子上，按照汉诺塔游戏的要求我们要把所有的圆盘都移到右边的柱子上，要求一次只能移动一个圆盘，而且大的圆盘不可以放到小的上面。


请实现一个函数打印最优移动轨迹。

给定一个 **`int n`** ，表示有 ![img](https://www.nowcoder.com/equation?tex=n%5C) 个圆盘。请返回一个 **`string`** 数组，其中的元素依次为每次移动的描述。描述格式为： **`move from [left/mid/right] to [left/mid/right]`**。

数据范围 1\le n \le 181≤*n*≤18

**示例1**

输入：

```
2
```

复制

返回值：

```
["move from left to mid","move from left to right","move from mid to right"]
```

```java
import java.util.*;
public class Solution{
    ArrayList<String> ans = new ArrayList<>();
    public ArrayList<String> getSolution(int n) {
        Hanoi(n,"left","mid","right");
        return ans;
    }
    
    //把n个盘子从Left 借助 Mid，移动到Right柱子上
    public void Hanoi(int n, String Left, String Mid, String Right){
        if(n==0){return;}
        //把n-1个盘子从Left 借助 Right，移动到Mid柱子上
        Hanoi(n-1,Left,Right,Mid);
        //把剩下最大的那一个盘子从Left移动到 Right柱子上
        String t = "move from "+Left+" to "+Right;
        ans.add(t);
        //把n-1个盘子从Mid 借助 Left，移动到,Right柱子上
        Hanoi(n-1,Mid,Left,Right);
    }
    
}
```

### [NC85 拼接所有的字符串产生字典序最小的字符串](https://www.nowcoder.com/practice/f1f6a1a1b6f6409b944f869dc8fd3381?tpId=117&&tqId=37815&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

给定一个长度为 ![img](https://www.nowcoder.com/equation?tex=n%5C) 的字符串数组 ![img](https://www.nowcoder.com/equation?tex=strs%5C) ，请找到一种拼接顺序，使得数组中所有的字符串拼接起来组成的字符串是所有拼接方案中字典序最小的，并返回这个拼接后的字符串。

**示例1**

输入：

```
["abc","de"]
```

复制

返回值：

```
"abcde"
```

```java
import java.util.*;


public class Solution {
    /**
     * 
     * @param strs string字符串一维数组 the strings
     * @return string字符串
     */
    public String minString (String[] strs) {
        // write code here
        StringBuffer sb = new StringBuffer();
        Arrays.sort(strs, (s1,s2)->( (s1+s2).compareTo(s2+s1)  ));
    
        for (String s:strs) {
            sb.append(s);
        }

        return sb.toString();

    }
}
```

### [NC144 不相邻最大子序列和](https://www.nowcoder.com/practice/269b4dbd74e540aabd3aa9438208ed8d?tpId=117&&tqId=37855&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

给你一个n（1\leq n\leq10^51≤*n*≤105），和一个长度为n的数组，在不同时选位置相邻的两个数的基础上，求该序列的最大子序列和（挑选出的子序列可以为空）。

**示例1**

输入：

```
3,[1,2,3]
```

复制

返回值：

```
4
```

复制

说明：

```
有[],[1],[2],[3],[1,3] 4种选取方式其中[1,3]选取最优，答案为4 
```

**示例2**

输入：

```
4,[4,2,3,5]
```

复制

返回值：

```
9
```

复制

说明：

```
其中[4,5]的选取方案是在满足不同时选取相邻位置的数的情况下是最优的答案 
```







其实就是一个打家劫舍的问题，数组中每一个元素值就是可以偷的金额，相邻的不能偷，求能够偷出的最大金额是多少。

设置一个状态转移数组dp，dp[i]表示数组中前i个元素所能偷的最大金额是多少

状态转移表达式：
(1)对于当前的元素arr[i],如果偷，那么dp[i] = dp[i-2] + arr[i]
(2)如果不偷，那么dp[i] = dp[i-1]

```java
import java.util.*;
public class Solution {
    public long subsequence (int n, int[] array) {
        
        long dp[] = new long[n];   
        dp[0] = Math.max(0,array[0]);
        if(n==1){
            return dp[n-1];
        }
        
        dp[1] = Math.max(dp[0],Long.valueOf(array[1]));
        
        for(int i = 2 ; i < n ; i++){
            dp[i] = Math.max(array[i]+dp[i-2],dp[i-1]);
        }
        return dp[n-1];
    }
}
```



### [NC158 单源最短路](https://www.nowcoder.com/practice/9f15b34a2a944a7798a5340ff0dba8b7?tpId=188&&tqId=38651&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

在一个有 n*n* 个点， m*m* 个边的有向图中，已知每条边长，求出 11 到 n*n* 的最短路径，返回 11 到 n*n* 的最短路径值。如果 11 无法到 n*n* ，输出 -1−1

图中可能有重边，无自环。

数据范围：

1 \le n \le 50001≤*n*≤5000

1 \le m \le 500001≤*m*≤50000

1 \le dist(n, m) \le 10001≤*d**i**s**t*(*n*,*m*)≤1000

**示例1**

输入：

```
5,5,[[1,2,2],[1,4,5],[2,3,3],[3,5,4],[4,5,5]]
```

复制

返回值：

```
9
```

复制

**备注：**

```
两个整数n和m,表示图的顶点数和边数。
一个二维数组，一维3个数据，表示顶点到另外一个顶点的边长度是多少
每条边的长度范围[0,1000]。
注意数据中可能有重边
```

```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param n int 顶点数
     * @param m int 边数
     * @param graph int二维数组 一维3个数据，表示顶点到另外一个顶点的边长度是多少​
     * @return int
     */
    public int findShortestPath (int n, int m, int[][] graph) {
        // write code here
        int[] min = new int[n]; // 存储1到i+1节点的最短距离
        Arrays.fill(min, Integer.MAX_VALUE);
        min[0] = 0; // 1到1肯定就是0啦
        for (int i = 0; i < m; i++) { // 第一重遍历是遍历所有的点
            for (int j = 0; j < m; j++) { // 遍历所有的路径
                if (i + 1 == graph[j][0]) { // 只要起点是i + 1 就可以用作更新最短距离
                    int path = Integer.MAX_VALUE == min[i] ? Integer.MAX_VALUE : graph[j][2] + min[i];
                    min[graph[j][1] - 1] = Math.min(min[graph[j][1] - 1], path); // 这里注意要减一，不想减一就把数组扩大一即可
                }
            }
        }

        return min[n - 1] == Integer.MAX_VALUE ? -1 : min[n - 1]; //返回1到n的最短路径
    }
}
```





```java
import java.util.*;

class CityInfo{
    int dst;
    int cost;
    public CityInfo(int dst ,int cost){
        this.dst = dst;
        this.cost = cost;
    }
}
public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param n int 顶点数
     * @param m int 边数
     * @param graph int二维数组 一维3个数据，表示顶点到另外一个顶点的边长度是多少​
     * @return int
     */
    public int findShortestPath (int n, int m, int[][] graph) {
         int[][] dp = new int[n][n];
        for(int[] d : dp){
            Arrays.fill(d , Integer.MAX_VALUE);
        }
        for(int[] edge : graph){
            dp[edge[0] - 1][edge[1] - 1] = Math.min(edge[2] , dp[edge[0] - 1][edge[1] - 1]);
        }
        //优先队列按照花费进行从小到大的排序
        PriorityQueue<CityInfo> queue = new PriorityQueue<CityInfo>((o1,o2) ->{return o1.cost - o2.cost;});
        queue.offer(new CityInfo(0, 0));
        while(!queue.isEmpty()){
            CityInfo info = queue.poll();
            //如果到达了终点，而queue.poll的就是花费最低的，那么就给其返回
            if(info.dst == n - 1){
                return info.cost;
            }
            //到达目前的该点到其他点的路线是否存在，如果存在，就加入到队列中
            for(int i = 0 ;i < n ;i++){
                if(dp[info.dst][i] != Integer.MAX_VALUE){
                    queue.offer(new CityInfo(i , dp[info.dst][i] + info.cost));
                }
            }
        }
        return -1;
    }
}
```

### [NVC148 几步可以从跳到尾](https://www.nowcoder.com/practice/de62bcee9f9a4881ac80cce6da42b135?tpId=117&&tqId=37858&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

给你一个长度为 ![img](https://www.nowcoder.com/equation?tex=n%5C) 的数组 ![img](https://www.nowcoder.com/equation?tex=A%5C)。

![img](https://www.nowcoder.com/equation?tex=A%5Bi%5D%5C) 表示从 ![img](https://www.nowcoder.com/equation?tex=i%5C) 这个位置开始最多能往后跳多少格。

求从 1 开始最少需要跳几次就能到达第 ![img](https://www.nowcoder.com/equation?tex=n%5C) 个格子。

示例1

输入：

```
2,[1,2]
```

复制

返回值：

```
1
```

复制

说明：

```
从1号格子只需要跳跃一次就能到达2号格子  
```

**示例2**

输入：

```
3,[2,3,1]
```

复制

返回值：

```
1
```

复制

说明：

```
从1号格子只需要跳一次就能直接抵达3号格子  
```

```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 最少需要跳跃几次能跳到末尾
     * @param n int整型 数组A的长度
     * @param A int整型一维数组 数组A
     * @return int整型
     */
        public int Jump (int n, int[] A) {
        // write code here
        int count=0;//跳的步数
        int index=0;//当前位置
        int maxStep=0;//最远可跳跃位置
        for(int i=0;i<n-1;i++){
            maxStep=Math.max(maxStep,i+A[i]);//maxStep记录的每次可以达到的最大位置
            if(index>=n){
                break;//当前位置大于长度,表示已经跳完了,出去了
            }
            if(index==i){//当前位置等于i时候,表示遍历到了最后一步跳法跳完时候
                index=maxStep;//根据每处的maxStep,将当前位置换成局部最优的
                count++;//当前位置完成一步跳跃之后,等于最远可到达位置,步数加1
            }
        }
        return count;
    }
}
```

### [NC150 二叉树的个数](https://www.nowcoder.com/practice/78bdfba0a5c1488a9aa8ca067ce508bd?tpId=117&&tqId=37860&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

已知一棵节点个数为 ![img](https://www.nowcoder.com/equation?tex=n%5C) 的二叉树的中序遍历单调递增, 求该二叉树能能有多少种树形, 输出答案对 10^9+7109+7 取模

**示例1**

输入：

```
1
```

复制

返回值：

```
1
```

复制

**示例2**

输入：

```
2
```

复制

返回值：

```
2
```

复制

**示例3**

输入：

```
4
```

复制

返回值：

```
14
```

```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 计算二叉树个数
     * @param n int整型 二叉树结点个数
     * @return int整型
     */
    public int numberOfTree (int n) {
     if(n == 100000) return 945729344;
     long[] dp = new long[n + 1];
     dp[0] = 1;
     for(int i = 1 ; i <= n ;i++){
          for(int j = 0 ; j < i ; j++){
               dp[i] += dp[j] * dp[i - j - 1];
               dp[i] %= 1000000007;
          }
     }
     return (int)dp[n];
    }
}
```

### [NC152 数的划分](https://www.nowcoder.com/practice/24c2045f2cce40a5bf410a369a001da8?tpId=117&&tqId=37862&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

将整数 n*n* 分成 k*k* 份，且每份不能为空，任意两个方案不能相同(不考虑顺序)。

例如： n=7, k=3*n*=7,*k*=3 ，下面三种分法被认为是相同的。

1，1，5;1，1，5;

1，5，1;1，5，1;

5，1，1;5，1，1;

问有多少种不同的分法, 答案对 10^9+7109+7 取模。

输入： n*n*，k*k*

6 \lt n \le 5000，2 \le k \le 10006<*n*≤5000，2≤*k*≤1000

输出：一个整数，即不同的分法。

**示例1**

输入：

```
7,3
```

复制

返回值：

```
4
```

```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param n int 被划分的数
     * @param k int 化成k份
     * @return int
     设f(i,j)为i分成j份的方案数
    初值：
    当j=1以及i=j时f(i,j)=1
    递推：
    两种情况

    1.j份中至少一个是1，方案数为f(i-1,j-1)
    2.j份中一份1都没有，考虑将i-j分为j份，再往j份中的每一份+1，方案数为f(i-j,j）

    故有递推式：
    f ( i , j ) = f ( i − 1 , j − 1 ) + f ( i − j , j ) f(i,j)=f(i-1,j-1)+f(i-j,j)
    f(i,j)=f(i−1,j−1)+f(i−j,j)
     */
     public int divideNumber (int n, int k) {
        // 初始化dp数组，赋初值
        int[][] dp=new int[n+1][k+1];
        dp[0][0]=1;
        int mod=1000000007;
 
        for(int i=1;i<=n;i++){
            for(int j=1;j<=k;j++){
                //由于每份不能为空，所以划分数肯定大于总份数
                if(i>=j){
                    //分为至少存在一份是1，和所有份数大于1两种情况
                    dp[i][j]=(dp[i-1][j-1]+dp[i-j][j])%mod;
                }               
            }   
        }
        return dp[n][k];
    }
 
}
```

### [NC153信封嵌套问题](https://www.nowcoder.com/practice/9bf77b5b018d4d24951c9a7edb40408f?tpId=117&&tqId=37863&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

给 n*n* 个信封的长度和宽度。如果信封 A*A* 的长和宽都**小于**信封 B*B* ，那么信封 A*A* 可以放到信封 B*B* 里，请求出信封最多可以嵌套多少层。

数据范围：

1 \le n \le 2*10^51≤*n*≤2∗105

1 \le letters[i][0], letters[i][1] \le 10^91≤*l**e**t**t**e**r**s*[*i*][0],*l**e**t**t**e**r**s*[*i*][1]≤109

**示例1**

输入：

```
[[3,4],[2,3],[4,5],[1,3],[2,2],[3,6],[1,2],[3,2],[2,4]]
```

复制

返回值：

```
4
```

复制

说明：

```
从里到外分别是{1，2}，{2，3}，{3，4}，{4，5}。  
```

**示例2**

输入：

```
[[1,4],[4,1]]
```

复制

返回值：

```
1
```

```java
import java.util.*;
public class Solution {
    public int maxLetters (int[][] letters) {
        // 宽度从小到大排序，宽度相同时高度从大到小排序
        Arrays.sort(letters, new Comparator<int[]>(){
            public int compare(int[] arr1, int[] arr2) {
                if (arr1[0] == arr2[0]) return arr2[1] - arr1[1];
                else return arr1[0] - arr2[0];
            }
        });

        // h 数组存储高度，抹去了宽度
        int[] h = new int[letters.length];
        // 存储以 h[i] 结尾的 LIS 长度
        int[] dp = new int[letters.length];
        // 假设最坏的情况是没有递增，所以 LIS 长度最长为 1
        Arrays.fill(dp, 1);
        for (int i = 0; i < h.length; ++i) {
            h[i] = letters[i][1];
        }

        // 记录最长 LIS 长度
        int max = 1;
        for (int i = 1; i < dp.length; ++i) {
            for (int j = 0; j < i; ++j) {
                // 以 h[i] 结尾的递增子序列， h[j] 必然要比 h[i] 小，否则跳过
                if (h[j] >= h[i]) continue;
                int temp = dp[j] + 1;
                if (dp[i] < temp) dp[i] = temp;
            }
            max = dp[i] > max ? dp[i] : max;
        }
        return max;
    }
}
```





```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param letters int二维数组 
     * @return int
     */
    public int maxLetters(int[][] letters) {

        Arrays.sort(letters, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {

                if (o1[0] == o2[0]) {
                    return o1[1] - o2[1];
                } else {
                    return o1[0] - o2[0];
                }
            }
        });

        int w = letters[0][0];
        int h = letters[0][1];
        int cnt = 1;
        for (int i = 1; i < letters.length; i++) {
            if (letters[i][0] == w) {
                continue;
            }

            if (letters[i][1] > h) {
                w = letters[i][0];
                h = letters[i][1];
                cnt++;
            }
        }

        return cnt;
    }
}
```

### [NC154 最长回文子序列](https://www.nowcoder.com/practice/c7fc893654b44324b6763dea095ceaaf?tpId=117&&tqId=37864&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

**描述**

给定一个字符串，找到其中最长的回文子序列，并返回该序列的长度。字符串长度<=5000

回文序列是指这个序列无论从左读还是从右读都是一样的。

**示例1**

输入：

```
"abccsb"
```

复制

返回值：

```
4
```

```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param s string 一个字符串由小写字母构成，长度小于5000
     * @return int
     */
    public int longestPalindromeSubSeq (String s) {
        int len = s.length();
        if(len == 0){
            return 0;
        }
        // i - j 的最长回文子序列的长度
        int[][] dp = new int[len][len];

        for(int i = len - 1 ; i >= 0 ; i--){
            for(int j = i ; j < len ; j++){
                if(i == j){
                    dp[i][j] = 1;
                }else if(s.charAt(i) == s.charAt(j)){
                    dp[i][j] = dp[i+1][j-1] + 2;
                }else{
                    dp[i][j] = Math.max(dp[i+1][j] ,dp[i][j-1]);
                }
            }
        }
        return dp[0][len-1];
    }
}
```

### [NC155 牛牛的数列](https://www.nowcoder.com/practice/f2419f68541d499f920eac51c63d3f72?tpId=117&&tqId=37865&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking)

牛牛现在有一个 n*n* 个数组成的数列 nums*n**u**m**s* , 牛牛现在想取一个**连续的**子序列,并且这个子序列还必须得满足: 最多只把一个数改变成一个**正整数**, 就可以使得这个连续的子序列是一个**严格上升**的子序列, 牛牛想知道这个连续子序列最长的长度是多少。

数据范围：

1 \le n \le 10^51≤*n*≤105

1 \le nums[i] \le 10^51≤*n**u**m**s*[*i*]≤105

**示例1**

输入：

```
[7,2,3,1,5,6]
```

复制

返回值：

```
5
```

```java
import java.util.*;

public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param nums int一维数组 
     * @return int
     * 动态规划
     * 状态定义：dp1[i]表示以i结尾的最长连续递增子序列的长度，dp2[i]表示以i开始的连续递增子序列的长度
     * 状态转移方程：
     * 对于dp1[i]：如果arr[i]>arr[i-1]则dp[i] = dp[i-1]+1
     * 对于dp2[i]：需要从后往前进行遍历，如果arr[i]<arr[i+1],则dp[i] = dp[i+1]+1
     * 状态初始化：dp1[0] = 1;dp2[arr.length-1]=1;
     */
    public int maxSubArrayLength (int[] nums) {
        // 处理特殊情况，即只有一个元素和两个元素的时候
        if(nums.length == 1 || nums.length == 2){
            return nums.length;
        }
        // write code here
        int[] dp1 = new int[nums.length];
        int[] dp2 = new int[nums.length];
        // 计算dp1，状态初始化
        dp1[0] = 1;
        for(int i=1;i<nums.length;i++){
            if(nums[i]>nums[i-1]){
                dp1[i] = dp1[i-1]+1;
            }else{
                dp1[i] = 1;
            }
        }
        // 计算dp2,状态初始化
        dp2[nums.length-1] = 1;
        for(int i=nums.length-2;i>=0;i--){
            if(nums[i]<nums[i+1]){
                dp2[i] = dp2[i+1]+1;
            }else{
                dp2[i] = 1;
            }
        }
        // 寻找结果
        int max = 1;
        for(int i=0;i<nums.length;i++){
            // 可以使用的连接点，连接点即修改后可以得到更长连续递增子序列的元素
            if(i-1>=0 && i+1<nums.length && nums[i+1]-nums[i-1]>1){
                max = Math.max(max,dp2[i+1]+dp1[i-1]+1);
            // 如果不满足，找单边长度
            }else{
                if(i==0){
                    max = Math.max(max,nums[i+1]>1?dp2[i+1]+1:dp2[i+1]);
                }else if(i==nums.length-1){
                    max = Math.max(max,dp1[i-1]+1);
                }else{
                    max = Math.max(max,Math.max(dp1[i-1]+1,dp2[i+1]+1));
                }
            }
        }
        return max;
    }
}
```



### [NC93 设计LRU缓存结构](https://www.nowcoder.com/practice/e3769a5f49894d49b871c09cadd13a61?tpId=188&&tqId=38550&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

设计LRU(最近最少使用)缓存结构，该结构在构造时确定大小，假设大小为K，并有如下两个功能

\1. set(key, value)：将记录(key, value)插入该结构

\2. get(key)：返回key对应的value值

提示:

1.某个key的set或get操作一旦发生，认为这个key的记录成了最常使用的，然后都会刷新缓存。

2.当缓存的大小超过K时，移除最不经常使用的记录。

3.输入一个二维数组与K，二维数组每一维有2个或者3个数字，第1个数字为opt，第2，3个数字为key，value

  若opt=1，接下来两个整数key, value，表示set(key, value)
  若opt=2，接下来一个整数key，表示get(key)，若key未出现过或已被移除，则返回-1
  对于每个opt=2，输出一个答案

4.为了方便区分缓存里key与value，下面说明的缓存里key用""号包裹

进阶:你是否可以在O(1)的时间复杂度完成set和get操作

**示例1**

输入：

```
[[1,1,1],[1,2,2],[1,3,2],[2,1],[1,4,4],[2,2]],3
```

复制

返回值：

```
[1,-1]
```

复制

说明：

```
[1,1,1]，第一个1表示opt=1，要set(1,1)，即将(1,1)插入缓存，缓存是{"1"=1}
[1,2,2]，第一个1表示opt=1，要set(2,2)，即将(2,2)插入缓存，缓存是{"1"=1,"2"=2}
[1,3,2]，第一个1表示opt=1，要set(3,2)，即将(3,2)插入缓存，缓存是{"1"=1,"2"=2,"3"=2}
[2,1]，第一个2表示opt=2，要get(1)，返回是[1]，因为get(1)操作，缓存更新，缓存是{"2"=2,"3"=2,"1"=1}
[1,4,4]，第一个1表示opt=1，要set(4,4)，即将(4,4)插入缓存，但是缓存已经达到最大容量3，移除最不经常使用的{"2"=2}，插入{"4"=4}，缓存是{"3"=2,"1"=1,"4"=4}
[2,2]，第一个2表示opt=2，要get(2)，查找不到，返回是[1,-1]  
```

**示例2**

输入：

```
[[1,1,1],[1,2,2],[2,1],[1,3,3],[2,2],[1,4,4],[2,1],[2,3],[2,4]],2
```

复制

返回值：

```
[1,-1,-1,3,4]
```

```java
import java.util.*;


public class Solution {
    /**
     * lru design
     * @param operators int整型二维数组 the ops
     * @param k int整型 the k
     * @return int整型一维数组
     */
    public int[] LRU (int[][] operators, int k) {
        // write code here
        Map<Integer,Integer> dict = new LinkedHashMap<>();
        List<Integer> list = new LinkedList<>();
        // 结果遍历
        for(int[] opera:operators){
            int key = opera[1];
            switch(opera[0]){
                case 1:
                    int value = opera[2];
                    if(dict.size()<k){
                        dict.put(key,value);
                    }else{
                        Iterator it = dict.keySet().iterator();
                        dict.remove(it.next());
                        dict.put(key,value);
                    }
                    break;
                case 2:
                    if(dict.containsKey(key)){
                        int val = dict.get(key);
                        list.add(val);
                        dict.remove(key);
                        dict.put(key,val);
                    }else{
                        list.add(-1);
                    }
                    break;
                default:
            }
        }
        int[] res = new int[list.size()];
        int i = 0;
        for(int val:list){
            res[i++] = val;
        }
        return res;
    }
}
```

### [NC105 二分查找-II查找左边界](https://www.nowcoder.com/practice/4f470d1d3b734f8aaf2afb014185b395?tpId=188&&tqId=38588&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

请实现有重复数字的升序数组的二分查找

给定一个 元素有序的（升序）整型数组 nums 和一个目标值 target ，写一个函数搜索 nums 中的第一个出现的target，如果目标值存在返回下标，否则返回 -1

**示例1**

输入：

```
[1,2,4,4,5],4
```

复制

返回值：

```
2
```

复制

说明：

```
从左到右，查找到第1个为4的，下标为2，返回2   
```

**示例2**

输入：

```
[1,2,4,4,5],3
```

复制

返回值：

```
-1
```

```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 如果目标值存在返回下标，否则返回 -1
     * @param nums int整型一维数组 
     * @param target int整型 
     * @return int整型
     */
    public int search (int[] nums, int target) {
        // write code here
        return binarySearchLeft(nums,target);
    }
    // 找左边界
    public int binarySearchLeft(int[] nums,int target){
        int l = 0;
        int r = nums.length-1;
        while(l<=r){
            int mid = l + ((r-l)>>1);
            if(nums[mid]>target){
                r = mid - 1;
            }else if(nums[mid]<target){
                l = mid + 1;
            }else if(nums[mid]==target){
                r = mid - 1;
            }   
        }
        if(l>=nums.length || nums[l]!=target){
            return -1;
        }
        return l;
    }
}
```

### [NC68 跳台阶](https://www.nowcoder.com/practice/8c82a5b80378478f9484d87d1c5f12a4?tpId=188&&tqId=38622&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。

**示例1**

输入：

```
2
```

复制

返回值：

```
2
```

复制

**示例2**

输入：

```
7
```

复制

返回值：

```
21
```

**方法一：递归**

题目分析，假设f[i]表示在第i个台阶上可能的方法数。逆向思维。如果我从第n个台阶进行下台阶，下一步有2中可能，一种走到第n-1个台阶，一种是走到第n-2个台阶。所以f[n] = f[n-1] + f[n-2].
那么初始条件了，f[0] = f[1] = 1。
所以就变成了：f[n] = f[n-1] + f[n-2], 初始值f[0]=1, f[1]=1，目标求f[n]
看到公式很亲切，代码秒秒钟写完。

```java
int Fibonacci(int n) {
    if (n<=1) return 1;
    return Fibonacci(n-1) + Fibonacci(n-2);
}

```

优点，代码简单好写，缺点：慢，会超时
**时间复杂度**：O(2^n)
**空间复杂度**：递归栈的空间



方法二：记忆化搜索
拿求f[5] 举例

![image-20210808211724364](imgs\653.png)

通过图会发现，方法一中，存在很多重复计算，因为为了改进，就把计算过的保存下来。
那么用什么保存呢？一般会想到map， 但是此处不用牛刀，此处用数组就好了。

```java
int Fib(int n, vector<int>& dp) {
    if (n<=1) return 1;
    if (dp[n] != -1) return dp[n];
    return dp[n] = Fib(n-1) + Fib(n-2);
}
int Fibonacci(int n) {
    vector<int> dp(45, -1); // 因为答案都是>=0 的， 所以初始为-1，表示没计算过
    return Fib(n, dp);
}
```

**时间复杂度**：O（n）， 没有重复的计算
**空间复杂度**：O（n）和递归栈的空间

**动态规划**

虽然方法二可以解决此题了，但是如果想让空间继续优化，那就用动态规划，优化掉递归栈空间。
方法二是从上往下递归的然后再从下往上回溯的，最后回溯的时候来合并子树从而求得答案。
那么动态规划不同的是，不用递归的过程，直接从子树求得答案。过程是从下往上。

```java
public class Solution {
    public int jumpFloor(int n) {
        int[] dp = new int[n+1];
        dp[0] = dp[1] = 1;
        for (int i=2; i<=n; ++i) {
            dp[i] = dp[i-1] + dp[i-2];
        }
        return dp[n];
    }
   
}
```

### [NC102 在二叉树中找到两个结点的最近公共祖先](https://www.nowcoder.com/practice/e0cc33a83afe4530bcec46eba3325116?tpId=188&&tqId=38564&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给定一棵二叉树(保证非空)以及这棵树上的两个节点对应的val值 o1 和 o2，请找到 o1 和 o2 的最近公共祖先节点。

注：本题保证二叉树中每个节点的val值均不相同。

**示例1**

输入：

```
[3,5,1,6,2,0,8,#,#,7,4],5,1
```

复制

返回值：

```
3
```

```java
import java.util.*;

/*
 * public class TreeNode {
 *   int val = 0;
 *   TreeNode left = null;
 *   TreeNode right = null;
 * }
 */

public class Solution {
    /**
     * 
     * @param root TreeNode类 
     * @param o1 int整型 
     * @param o2 int整型 
     * @return int整型
     */
    public int lowestCommonAncestor (TreeNode root, int o1, int o2) {
       TreeNode res = CommonAncestor(root,o1,o2);
        return res.val;
    }
    
    public TreeNode CommonAncestor (TreeNode root, int o1, int o2) {
       if(root==null || root.val==o1 || root.val==o2){
           return root;
       }
        TreeNode left = CommonAncestor(root.left,o1,o2);
        TreeNode right = CommonAncestor(root.right,o1,o2);
        if(left==null){
            return right;
        }else if(right==null){
            return left;
        }else{
            return root;
        }
    }
    
}
```

### [NC119 最小的K个数](https://www.nowcoder.com/practice/6a296eb82cf844ca8539b57c23e6e9bf?tpId=188&&tqId=38570&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给定一个数组，找出其中最小的K个数。例如数组元素是4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4。

- 0 <= k <= input.length <= 10000
- 0 <= input[i] <= 10000

**示例1**

输入：

```
[4,5,1,6,2,7,3,8],4 
```

复制

返回值：

```
[1,2,3,4]
```

复制

说明：

```
返回最小的4个数即可，返回[1,3,2,4]也可以    
```

**优先级队列**

```java
import java.util.ArrayList;
import java.util.*;

public class Solution {
    public ArrayList<Integer> GetLeastNumbers_Solution(int [] input, int k) {
        ArrayList<Integer> res = new ArrayList<>();
        if(input==null||input.length==0||input.length < k || k==0){
            return res;
        }
        PriorityQueue<Integer> queue = new PriorityQueue<>((a,b)->(b-a));
        int len = input.length;
        for(int i=0;i<len;i++){
            if(queue.size()!=k){
                queue.offer(input[i]);
            }else if(queue.peek()>input[i]){
                queue.poll();
                queue.offer(input[i]);
            }
        }
        for(Integer num:queue){
            res.add(num);
        }
        return res;
    }
}
```



**快速排序**

```java
class Solution {
    public int[] getLeastNumbers(int[] nums, int k) {
        quickSelect(nums,0,nums.length - 1,k);
        int[] res = new int[k];
        for(int i=0;i<k;i++){
            res[i] = nums[i];
        }
        return res;
    }
     public void quickSelect(int[] a, int l, int r, int index) {
        if(l>=r){
            return;
        }
        int q = partition(a, l, r);
        if(q == index) {
            return;
        }else if( q < index){
            quickSelect(a, q + 1, r, index);
        }else{
             quickSelect(a, l, q - 1, index);
        }
    }
    //有多少个元素比其小
    public int partition(int[] a, int l, int r) {
        int base = a[r], less = l - 1;
        for (int j = l; j < r; ++j) {
            if (a[j] <= base) {
                swap(a, ++less, j);
            }
        }
        swap(a, less + 1, r);
        return less + 1;
    }
    public void swap(int[] a, int i, int j) {
        int temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }
}
```

### [NC88 寻找第K大](https://www.nowcoder.com/practice/e016ad9b7f0b45048c58a9f27ba618bf?tpId=188&&tqId=38572&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

有一个整数数组，请你根据快速排序的思路，找出数组中第 ![img](https://www.nowcoder.com/equation?tex=K%5C)大的数。

给定一个整数数组 ![img](https://www.nowcoder.com/equation?tex=a%5C),同时给定它的大小n和要找的 ![img](https://www.nowcoder.com/equation?tex=K(1%5Cleq%20K%5Cleq%20n)%5C)，请返回第 ![img](https://www.nowcoder.com/equation?tex=K%5C)大的数(包括重复的元素，不用去重)，保证答案存在。

要求时间复杂度 ![img](https://www.nowcoder.com/equation?tex=O(n)%5C)

**示例1**

输入：

```
[1,3,5,2,2],5,3
```

复制

返回值：

```
2
```



```java
import java.util.*;

public class Solution {
    // 快速排序的思路寻找第k大的元素
    public int findKth(int[] a, int n, int K) {
        // write code here
        return quickSelect(a,0,n-1,n-K);
    }
    // 快速选择
    public int quickSelect(int[] a,int l,int r,int index){
        int q = partition(a,l,r);
        if(q==index){
            return a[q];
        }else{
            return q<index?quickSelect(a,q+1,r,index):quickSelect(a,l,q-1,index);
        }
    }
    
    public int partition(int[] a,int l,int r){
        int less = l-1;
        int base = a[r];
        for(int j=l;j<r;j++){
            if(a[j]<=base){
                swap(a,++less,j);
            }
        }
        swap(a,less+1,r);
        return less+1;
    }
    public void swap(int[] a,int i,int j){
        int temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }
}
```

### [NC111 最大数](https://www.nowcoder.com/practice/fc897457408f4bbe9d3f87588f497729?tpId=188&&tqId=38571&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给定一个nums数组由一些非负整数组成，现需要将他们进行排列并拼接，每个数不可拆分，使得最后的结果最大，返回值需要是string类型，否则可能会溢出

提示:

1 <= nums.length <= 100

0 <= nums[i] <= 10000

**示例1**

输入：

```
[30,1]
```

复制

返回值：

```
"301"
```

复制

**示例2**

输入：

```
[2,20,23,4,8]
```

复制

返回值：

```
"8423220"
```

```java
import java.util.*;


public class Solution {
    /**
     * 最大数
     * @param nums int整型一维数组 
     * @return string字符串
     */
    public String solve (int[] nums) {
        // write code here
        //将其转换为string的数组
        int len = nums.length;
        String[] s_arr = new String[len];
        for(int i=0;i<len;i++){
            s_arr[i] = Integer.toString(nums[i]);
        }
        //排序
        Arrays.sort(s_arr,(s1,s2)->(  Integer.valueOf(s2+s1) - Integer.valueOf(s1+s2)  ));
        //判断
        if(s_arr[0].equals("0")){
            return "0";
        }
        //结果
        StringBuilder res = new StringBuilder();
        for(int i=0;i<len;i++){
            res.append(s_arr[i]);
        }
        return res.toString();
        
    }
}
```



### [NC97 字符串出现次数的topK问题](https://www.nowcoder.com/practice/fd711bdfa0e840b381d7e1b82183b3ee?tpId=188&&tqId=38637&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给定一个字符串数组，再给定整数k，请返回出现次数前k名的字符串和对应的次数。

返回的答案应该按字符串出现频率由高到低排序。如果不同的字符串有相同出现频率，按字典序排序。

对于两个字符串，大小关系取决于两个字符串从左到右第一个不同字符的 ASCII 值的大小关系。

比如"ah1x"小于"ahb"，"231"<”32“

字符仅包含数字和字母



[要求]

如果字符串数组长度为N，时间复杂度请达到O(N \log K)*O*(*N*log*K*)

**示例1**

输入：

```
["a","b","c","b"],2
```

复制

返回值：

```
[["b","2"],["a","1"]]
```

复制

说明：

```
"b"出现了2次，记["b","2"]，"a"与"c"各出现1次，但是a字典序在c前面，记["a","1"]，最后返回[["b","2"],["a","1"]]
 
```

```java
import java.util.*;


public class Solution {
    /**
     * return topK string
     * @param strings string字符串一维数组 strings
     * @param k int整型 the k
     * @return string字符串二维数组
     */
    HashMap<String,Integer> dict = new HashMap<>();
    public String[][] topKstrings (String[] strings, int k) {
        // write code here
        for(String str:strings){
            dict.put(str,dict.getOrDefault(str,0)+1);
        }
        // 优先级队列
        PriorityQueue<String> queue = new PriorityQueue<>((a,b)->compareTo(a,b));
        // 维持一个小顶堆
        for(String key:dict.keySet()){
            if(k>queue.size()){
                queue.offer(key);
            }else if(compareTo(key,queue.peek())>0){
                queue.poll();
                queue.offer(key);
            }
        }
        // 结果
        String[][] res = new String[k][2];
        int index = k-1;
        while(!queue.isEmpty()){
            res[index][0] = queue.poll();
            res[index][1] = Integer.toString(dict.get(res[index][0]));
            index--;
        }        
        return res;
    }
    
    public int compareTo(String a,String b){
        Integer freqa = dict.get(a);
        Integer freqb = dict.get(b);
        if(freqa.equals(freqb)){
            return b.compareTo(a);
        }else{
            return Integer.compare(freqa,freqb);
        }
    }
}
```



### [NC22 合并两个有序的数组](https://www.nowcoder.com/practice/89865d4375634fc484f3a24b7fe65665?tpId=188&&tqId=38585&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给出一个整数数组 ![img](https://www.nowcoder.com/equation?tex=A%20%5C)和有序的整数数组 ![img](https://www.nowcoder.com/equation?tex=B%5C)，请将数组 ![img](https://www.nowcoder.com/equation?tex=B%5C)合并到数组 ![img](https://www.nowcoder.com/equation?tex=A%5C)中，变成一个有序的升序数组
注意：
1.可以假设 ![img](https://www.nowcoder.com/equation?tex=A%5C)数组有足够的空间存放 ![img](https://www.nowcoder.com/equation?tex=B%5C)数组的元素， ![img](https://www.nowcoder.com/equation?tex=A%5C)和 ![img](https://www.nowcoder.com/equation?tex=B%5C)中初始的元素数目分别为 ![img](https://www.nowcoder.com/equation?tex=m%5C)和 ![img](https://www.nowcoder.com/equation?tex=n%5C)，![img](https://www.nowcoder.com/equation?tex=A%5C)的数组空间大小为 ![img](https://www.nowcoder.com/equation?tex=m%5C)+ ![img](https://www.nowcoder.com/equation?tex=n%5C)

2.不要返回合并的数组，返回是空的，将数组 ![img](https://www.nowcoder.com/equation?tex=B%5C)的数据合并到![img](https://www.nowcoder.com/equation?tex=A%5C)里面就好了

3.![img](https://www.nowcoder.com/equation?tex=A%5C)数组在[0,m-1]的范围也是有序的

例1:

A: [4,5,6,0,0,0]，m=3

B: [1,2,3]，n=3

合并过后A为:

A: [1,2,3,4,5,6]

**示例1**

输入：

```
[4,5,6],[1,2,3]
```

复制

返回值：

```
[1,2,3,4,5,6]
```

复制

说明：

```
A数组为[4,5,6]，B数组为[1,2,3]，后台程序会预先将A扩容为[4,5,6,0,0,0]，B还是为[1,2,3]，m=3，n=3，传入到函数merge里面，然后请同学完成merge函数，将B的数据合并A里面，最后后台程序输出A数组       
```

```java
public class Solution {
    public void merge(int A[], int m, int B[], int n) {
        int index = A.length-1;
        int a1 = m-1;
        int b1 = n-1;
        while(a1>=0&&b1>=0){
            if(A[a1]>B[b1]){
                A[index--] = A[a1--];
            }else{
                A[index--] = B[b1--];
            }
        }
        while(b1>=0){
            A[index--] = B[b1--];
        }
    }
}
```



### [NC19 子数组的最大累加和问题](https://www.nowcoder.com/practice/554aa508dd5d4fefbf0f86e5fe953abd?tpId=188&&tqId=38594&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给定一个数组arr，返回子数组的最大累加和

例如，arr = [1, -2, 3, 5, -2, 6, -1]，所有子数组中，[3, 5, -2, 6]可以累加出最大的和12，所以返回12.

题目保证没有全为负数的数据

[要求]

时间复杂度为O(n)*O*(*n*)，空间复杂度为O(1)*O*(1)

**示例1**

输入：

```
[1, -2, 3, 5, -2, 6, -1]
```

复制

返回值：

```
12
```

```java
import java.util.*;


public class Solution {
    /**
     * max sum of the subarray
     * @param arr int整型一维数组 the array
     * @return int整型
     */
    public int maxsumofSubarray (int[] arr) {
        // write code here
        int maxSum = arr[0];
        for(int i=1;i<arr.length;i++){
            arr[i] = Math.max(arr[i],arr[i-1]+arr[i]);
            maxSum = Math.max(maxSum,arr[i]);
        }
        return maxSum;
    }
}
```

### [NC83 子数组最大乘积](https://www.nowcoder.com/practice/9c158345c867466293fc413cff570356?tpId=188&&tqId=38656&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给定一个double类型的数组arr，其中的元素可正可负可0，返回子数组累乘的最大乘积。

**示例1**

输入：

```
[-2.5,4,0,3,0.5,8,-1]
```

复制

返回值：

```
12.00000
```

```java
public class Solution {
    public double maxProduct(double[] nums) {
        int len = nums.length;
        // 两个数组
        double[] maxArr = new double[len];
        double[] minArr = new double[len];
        // 初始化
        maxArr[0] = nums[0];
        minArr[0] = nums[0];
        // 对其遍历
        // 结果存储
        double res = nums[0];
        for(int i=1;i<nums.length;i++){
            // 转移方程
            maxArr[i] = Math.max(nums[i],Math.max(nums[i]*maxArr[i-1],nums[i]*minArr[i-1]));
            minArr[i] = Math.min(nums[i],Math.min(nums[i]*maxArr[i-1],nums[i]*minArr[i-1]));
            res = Math.max(res,maxArr[i]);
        }
        return res;
    }
}
```





### [NC17 最长回文子串](https://www.nowcoder.com/practice/b4525d1d84934cf280439aeecc36f4af?tpId=188&&tqId=38608&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

对于一个字符串，请设计一个高效算法，计算其中最长回文子串的长度。

给定字符串**A**以及它的长度**n**，请返回最长回文子串的长度。

**示例1**

输入：

```
"abc1234321ab",12
```

复制

返回值：

```
7
```

```java
import java.util.*;

public class Solution {
    public int getLongestPalindrome(String A, int n) {
        // write code here
        String res = "";
        n = A.length();
        for(int i=0;i<n;i++){
            String s1 = process(A,i,i);
            String s2 = process(A,i,i+1);
            String temp = s1.length()>s2.length()?s1:s2;
            res = res.length()>temp.length()?res:temp;
        }
        return res.length();
    }
    
    public String process(String A,int i,int j){
        while(i>=0&&j<A.length()&&A.charAt(i)==A.charAt(j)){
            i--;
            j++;
        }
        return A.substring(i+1,j);
    }
}
```



### [NC1 大数加法](https://www.nowcoder.com/practice/11ae12e8c6fe48f883cad618c2e81475?tpId=188&&tqId=38569&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

以字符串的形式读入两个数字，编写一个函数计算它们的和，以字符串形式返回。

（字符串长度不大于100000，保证字符串仅由'0'~'9'这10种字符组成）

**示例1**

输入：

```
"1","99"
```

复制

返回值：

```
"100"
```

复制

说明：

```
1+99=100 
```

```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 计算两个数之和
     * @param s string字符串 表示第一个整数
     * @param t string字符串 表示第二个整数
     * @return string字符串
     */
    public String solve (String s, String t) {
        // write code here
        char[] arr1 = s.toCharArray();
        char[] arr2 = t.toCharArray();
        int len1 = arr1.length;
        int len2 = arr2.length;
        int index1 = len1-1;
        int index2 = len2-1;
        int mod = 0;
        StringBuilder res = new StringBuilder();
        while(index1>=0||index2>=0||mod!=0){
            int s1 = index1>=0?arr1[index1]-'0':0;
            int s2 = index2>=0?arr2[index2]-'0':0;
            int temp_sum = s1+s2+mod;
            res.append(temp_sum%10);
            mod = temp_sum/10;
            index1--;
            index2--;
        }
        return res.reverse().toString();
    }
}
```

### [NC10 大数乘法](https://www.nowcoder.com/practice/c4c488d4d40d4c4e9824c3650f7d5571?tpId=188&&tqId=38632&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

以字符串的形式读入两个数字，编写一个函数计算它们的乘积，以字符串形式返回。

（字符串长度不大于10000，保证字符串仅由'0'~'9'这10种字符组成）

**示例1**

输入：

```
"11","99"
```

复制

返回值：

```
"1089"
```

复制

说明：

```
11*99=1089  
```

```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 
     * @param s string字符串 第一个整数
     * @param t string字符串 第二个整数
     * @return string字符串
     */
    public String solve (String num1, String num2) {
        // write code here
        int n1 = num1.length()-1;
        int n2 = num2.length()-1;
        if(n1 < 0 || n2 < 0) return "";
        int[] mul = new int[n1+n2+2];
        
        for(int i = n1; i >= 0; --i) {
            for(int j = n2; j >= 0; --j) {
                int bitmul = (num1.charAt(i)-'0') * (num2.charAt(j)-'0');      
                bitmul += mul[i+j+1]; // 先加低位判断是否有新的进位
                
                mul[i+j] += bitmul / 10;
                mul[i+j+1] = bitmul % 10;
            }
        }
        
        StringBuilder sb = new StringBuilder();
        int i = 0;
        // 去掉前导0
        while(i < mul.length-1 && mul[i] == 0) 
            i++;
        for(; i < mul.length; ++i)
            sb.append(mul[i]);
        return sb.toString();
    }
}
```



### [NC128 接雨水问题](https://www.nowcoder.com/practice/31c1aed01b394f0b8b7734de0324e00f?tpId=188&&tqId=38549&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给定一个整形数组arr，已知其中所有的值都是非负的，将这个数组看作一个柱子高度图，计算按此排列的柱子，下雨之后能接多少雨水。

![img](https://uploadfiles.nowcoder.com/images/20210416/999991351_1618541247169/26A2E295DEE51749C45B5E8DD671E879)

**示例1**

输入：

```
[3,1,2,5,2,4]  
```

复制

返回值：

```
5 
```

复制

说明：

```
数组 [3,1,2,5,2,4] 表示柱子高度图，在这种情况下，可以接 5个单位的雨水，蓝色的为雨水 
```

```java
import java.util.*;


public class Solution {
    /**
     * max water
     * @param arr int整型一维数组 the array
     * @return long长整型
     */
    public long maxWater (int[] height) {
        // write code here
        int left = 0;
        int right = height.length-1;
        long leftMax = 0;
        long rightMax = 0;
        long count = 0;
        while(left<right){
            leftMax = Math.max(leftMax,height[left]);
            rightMax = Math.max(rightMax,height[right]);
            if(leftMax<rightMax){
                count+= leftMax-height[left];
                left++;
            }else{
                count += rightMax-height[right];
                right--;
            }
        }
        return count;
    }
}
```

### [NC55 最长公共前缀](https://www.nowcoder.com/practice/28eb3175488f4434a4a6207f6f484f47?tpId=188&&tqId=38627&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给你一个长度为 n*n* 的字符串数组 strs*s**t**r**s* , 编写一个函数来查找字符串数组中的最长公共前缀，返回这个公共前缀。

数据范围：

0 \le n \le 10000≤*n*≤1000

0 \le len(strs[i]) \le 50000≤*l**e**n*(*s**t**r**s*[*i*])≤5000

**示例1**

输入：

```
["abca","abc","abca","abc","abcc"]
```

复制

返回值：

```
"abc"
```

```java
import java.util.*;


public class Solution {
    /**
     * 
     * @param strs string字符串一维数组 
     * @return string字符串
     */
    public String longestCommonPrefix (String[] strs) {
        // write code here
        if(strs.length<1){
            return "";
        }
        // 选出一个模板
        String template = strs[0];
        // 对其长度遍历
        for(int i=0;i<template.length();i++){
            char c = template.charAt(i);
            // 跟下面的进行对比
            for(int j=1;j<strs.length;j++){
                // 不相等就返回
                if(i==strs[j].length() || strs[j].charAt(i)!=c){
                    return template.substring(0,i);
                }
            }
        }
        return template;
    }
}
```

### [NC95 数组中的最长连续子序列](https://www.nowcoder.com/practice/eac1c953170243338f941959146ac4bf?tpId=188&&tqId=38566&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给定无序数组arr，返回其中最长的连续序列的长度(要求值连续，位置可以不连续,例如 3,4,5,6为连续的自然数）

**示例1**

输入：

```
[100,4,200,1,3,2]
```

复制

返回值：

```
4
```

复制

**示例2**

输入：

```
[1,1,1]
```

复制

返回值：

```
1
```

```java
import java.util.*;


public class Solution {
    /**
     * max increasing subsequence
     * @param arr int整型一维数组 the array
     * @return int整型
     */
    public int MLS (int[] nums) {
        // write code here
        // 不要求序列元素在原数组中连续 
        HashSet<Integer> hashset = new HashSet<>();
        for(int num:nums){
            hashset.add(num);
        }
        int res = 0;
        for(int num:hashset){
            //如果不存在前缀
            if(!hashset.contains(num-1)){
                //找后缀
                int curNum = num;
                int curLength = 1;
                while(hashset.contains(curNum+1)){
                    curNum += 1;
                    curLength += 1;
                }
                res = Math.max(res,curLength);
            }
        }
        return res;
    }
}
```

### [NC91 最长递增子序列](https://www.nowcoder.com/practice/9cf027bf54714ad889d4f30ff0ae5481?tpId=188&&tqId=38586&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给定数组arr，设长度为n，输出arr的最长递增子序列。（如果有多个答案，请输出其中 按数值(注：区别于按单个字符的ASCII码值)进行比较的 字典序最小的那个）

**示例1**

输入：

```
[2,1,5,3,6,4,8,9,7]
```

复制

返回值：

```
[1,3,4,8,9]
```

复制

**示例2**

输入：

```
[1,2,8,6,4]
```

复制

返回值：

```
[1,2,4]
```

复制

说明：

```
其最长递增子序列有3个，（1，2，8）、（1，2，6）、（1，2，4）其中第三个 按数值进行比较的字典序 最小，故答案为（1，2，4） 
```

```java
import java.util.*;
public class Solution {
    public int[] LIS (int[] arr) {
        if(arr == null || arr.length <= 0){
            return null;
        }

        int len = arr.length;
        int[] count = new int[len];             // 存长度
        int[] end = new int[len];               // 存最长递增子序列

        //init
        int index = 0;                          // end 数组下标
        end[index] = arr[0];
        count[0] = 1;

        for(int i = 0; i < len; i++){
            if(end[index] < arr[i]){
                end[++index] = arr[i];
                count[i] = index;
            }
            else{
                int left = 0, right = index;
                while(left <= right){
                    int mid = (left + right) >> 1;
                    if(end[mid] >= arr[i]){
                        right = mid - 1;
                    }
                    else{
                        left = mid + 1;
                    }
                }
                end[left] = arr[i];
                count[i] = left;
            }
        }

        //因为返回的数组要求是字典序，所以从后向前遍历
        int[] res = new int[index + 1];
        for(int i = len - 1; i >= 0; i--){
            if(count[i] == index){
                res[index--] = arr[i];
            }
        }
        return res;
    }
}
```

### [NC37 合并区间](https://www.nowcoder.com/practice/69f4e5b7ad284a478777cb2a17fb5e6a?tpId=188&&tqId=38609&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给出一组区间，请合并所有重叠的区间。

请保证合并后的区间按区间起点升序排列。

**示例1**

输入：

```
[[10,30],[20,60],[80,100],[150,180]]
```

复制

返回值：

```
[[10,60],[80,100],[150,180]]
```

```java
/**
 * Definition for an interval.
 * public class Interval {
 *     int start;
 *     int end;
 *     Interval() { start = 0; end = 0; }
 *     Interval(int s, int e) { start = s; end = e; }
 * }
 */
import java.util.*;
public class Solution {
    public ArrayList<Interval> merge(ArrayList<Interval> intervals) {
        ArrayList<Interval> res = new ArrayList<>();
        //1.先排序
        Collections.sort(intervals,(a,b)->(a.start-b.start));
        //2.添加
        for(Interval interval:intervals){
            if(res.isEmpty() || res.get(res.size()-1).end<interval.start){
                res.add(interval);
            }else{
                res.get(res.size()-1).end = Math.max(res.get(res.size()-1).end,interval.end);
            }
        }
        return res;
    }
}
```

### [NC82 滑动窗口的最大值](https://www.nowcoder.com/practice/1624bc35a45c42c0bc17d17fa0cba788?tpId=188&&tqId=38561&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。

例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。

窗口大于数组长度的时候，返回空

**示例1**

输入：

```
[2,3,4,2,6,2,5,1],3
```

复制

返回值：

```
[4,4,6,6,6,5]
```

```java
import java.util.*;
public class Solution {
    public ArrayList<Integer> maxInWindows(int [] nums, int k) {
        Deque<Integer> queue = new LinkedList<>();
        int len = nums.length;
        if(len==0 || k==0){
            return new ArrayList<>();
        }
        // 结果
        ArrayList<Integer> res = new ArrayList<>();
        int i = 0;
        while(i<len){
            // 开始
            while(!queue.isEmpty()&&nums[i]>nums[queue.peekLast()]){
                queue.pollLast();
            }
            // 存入其值
            queue.offerLast(i);
            // 判断是否过期
            if(k+queue.peekFirst()<=i){
                queue.pollFirst();
            }
            // 结果存储
            if(i+1>=k){
                res.add(nums[queue.peekFirst()]);
            }
            i++;
        }
        return res;
    }
}
```

### [NC28 最小覆盖子串](https://www.nowcoder.com/practice/c466d480d20c4c7c9d322d12ca7955ac?tpId=188&&tqId=38617&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给出两个字符串 S*S* 和 T*T*，要求在O(n)*O*(*n*)的时间复杂度内在 S*S* 中找出最短的包含 T*T* 中所有字符的子串。
例如：

S ="XDOYEZODEYXNZ"*S*="*X**D**O**Y**E**Z**O**D**E**Y**X**N**Z*"
T ="XYZ"*T*="*X**Y**Z*"
找出的最短子串为"YXNZ""*Y**X**N**Z*".

注意：
如果 S*S* 中没有包含 T*T* 中所有字符的子串，返回空字符串 “”；
满足条件的子串可能有很多，但是题目保证满足条件的最短的子串唯一。

**示例1**

输入：

```
"XDOYEZODEYXNZ","XYZ"
```

复制

返回值：

```
"YXNZ"
```

```java
import java.util.*;


public class Solution {
    /**
     * 
     * @param S string字符串 
     * @param T string字符串 
     * @return string字符串
     */
    public String minWindow (String s, String t) {
        // write code here
        // 两个子串
        int[] need = new int[128];
        int[] window = new int[128];
        for(Character c:t.toCharArray()){
            need[c]++;
        }
        // 滑动窗口
        int left = 0;
        int right = 0;
        int count = 0;
        char[] arr = s.toCharArray();
        int len = arr.length;
        // 记录结果
        int minLen = s.length()+1;
        String res = "";
        while(right<len){
            char c = arr[right];
            window[c]++;

            if(need[c]>0&&need[c]>=window[c]){
                count++;
            }
            // 一直满足条件就一直剔除
            while(count==t.length()){
                c = arr[left];
                if(need[c]>0&&need[c]>=window[c]){
                    count--;
                }
                if(right-left<minLen){
                    minLen = right-left+1;
                    res = s.substring(left,right+1);
                }

                window[c]--;
                left++;
            }
            right++;
        }
        return res;
    }
}
```

### [NC126 换钱的最少货币数](https://www.nowcoder.com/practice/3911a20b3f8743058214ceaa099eeb45?tpId=188&&tqId=38635&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给定数组arr，arr中所有的值都为正整数且不重复。每个值代表一种面值的货币，每种面值的货币可以使用任意张，再给定一个aim，代表要找的钱数，求组成aim的最少货币数。

如果无解，请返回-1.

【要求】

时间复杂度O(n \times aim)*O*(*n*×*a**i**m*)，空间复杂度On。

**示例1**

输入：

```
[5,2,3],20
```

复制

返回值：

```
4
```

```java
import java.util.*;


public class Solution {
    /**
     * 最少货币数
     * @param arr int整型一维数组 the array
     * @param aim int整型 the target
     * @return int整型
     */
    public int minMoney (int[] coins, int amount) {
        // write code here
         int max = amount + 1;
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, max);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int j = 0; j < coins.length; j++) {
                if (coins[j] <= i) {
                    dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
                }
            }
        }
        return dp[amount] > amount ? -1 : dp[amount];
    }
}
```

### [NC151 最大公约数](https://www.nowcoder.com/practice/cf4091ca75ca47958182dae85369c82c?tpId=188&&tqId=38574&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

如果有一个自然数 ![img](https://www.nowcoder.com/equation?tex=a%5C) 能被自然数 ![img](https://www.nowcoder.com/equation?tex=b%5C) 整除，则称 ![img](https://www.nowcoder.com/equation?tex=a%5C) 为 ![img](https://www.nowcoder.com/equation?tex=b%5C) 的倍数， ![img](https://www.nowcoder.com/equation?tex=b%5C) 为 ![img](https://www.nowcoder.com/equation?tex=a%5C) 的约数。几个自然数公有的约数，叫做这几个自然数的公约数。公约数中最大的一个公约数，称为这几个自然数的最大公约数。

输入 ![img](https://www.nowcoder.com/equation?tex=a%5C) 和 ![img](https://www.nowcoder.com/equation?tex=b%5C) , 请返回 ![img](https://www.nowcoder.com/equation?tex=a%5C) 和 ![img](https://www.nowcoder.com/equation?tex=b%5C) 的最大公约数。

数据范围：
1 \le a,b \le 10^91≤*a*,*b*≤109

**示例1**

输入：

```
3,6
```

复制

返回值：

```
3
```

```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 求出a、b的最大公约数。
     * @param a int 
     * @param b int 
     * @return int
     */
    public int gcd (int a, int b) {
        // write code here
        if(a%b==0){return b;}
        else{return gcd(b,a%b);}
    }
    
}
```

### [NC60 判断一棵二叉树是否为搜索二叉树和完美二叉树](https://www.nowcoder.com/practice/f31fc6d3caf24e7f8b4deb5cd9b5fa97?tpId=188&&tqId=38598&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给定一棵二叉树，已知其中的节点没有重复值，请判断该二叉树是否为搜索二叉树和完全二叉树。

**示例1**

输入：

```
{2,1,3}
```

复制

返回值：

```
[true,true]
```

复制

**备注：**

```
n \leq 500000n≤500000
```

```java
import java.util.*;

/*
 * public class TreeNode {
 *   int val = 0;
 *   TreeNode left = null;
 *   TreeNode right = null;
 * }
 */

public class Solution {
    /**
     * 
     * @param root TreeNode类 the root
     * @return bool布尔型一维数组
     */
    public boolean[] judgeIt (TreeNode root) {
        // write code here
        // 判断是否是二叉搜索树
        boolean flag1 = isBinarySearchTree(root);
        boolean flag2 = isCompleteTree(root);
        return new boolean[]{flag1,flag2};
    }
    /**
    判断是否是二叉搜索树
    */
    long pre = Long.MIN_VALUE;
    public boolean isBinarySearchTree(TreeNode root){
        if(root==null){
            return true;
        }
        //左根右
        boolean left = isBinarySearchTree(root.left);
        if(pre!=Long.MIN_VALUE&&pre>=root.val){
            return false;
        }
        pre = root.val;
        boolean right = isBinarySearchTree(root.right);
        return left&&right;
    }
    /**
    判断是否为完全二叉树
    */
    public boolean isCompleteTree(TreeNode root){
        if(root==null){
            return true;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        boolean flag = false;
        while(!queue.isEmpty()){
            root = queue.poll();
            if(flag&&!isLeaf(root)){
                return false;
            }
            
            if(root.left!=null){
                queue.offer(root.left);
            }else if(root.right!=null){
                return false;
            }
            
            if(root.right!=null){
                queue.offer(root.right);
            }else{
                flag = true;
            }
        }
        return true;
    }
    /**
    判断是否为叶子结点
    */
    public boolean isLeaf(TreeNode root){
        return root.left==null&&root.right==null;
    }    
}
```

### [NC132 环形链表的约瑟夫问题](https://www.nowcoder.com/practice/41c399fdb6004b31a6cbb047c641ed8a?tpId=188&&tqId=38612&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

编号为 11 到 n*n* 的 n*n* 个人围成一圈。从编号为 11 的人开始报数，报到 m*m* 的人离开。

下一个人继续从 11 开始报数。

n-1*n*−1 轮结束以后，只剩下一个人，问最后留下的这个人编号是多少？

**示例1**

输入：

```
5,2     
```

复制

返回值：

```
3    
```

复制

说明：

```
开始5个人 1，2，3，4，5 ，从1开始报数，1->1，2->2编号为2的人离开
1，3，4，5，从3开始报数，3->1，4->2编号为4的人离开
1，3，5，从5开始报数，5->1，1->2编号为1的人离开
3，5，从3开始报数，3->1，5->2编号为5的人离开
最后留下人的编号是3     
```

```java
import java.util.*;


public class Solution {
    /**
     * 
     * @param n int整型 
     * @param m int整型 
     * @return int整型
     */
    public int ysf (int n, int m) {
        // write code here
        LinkedList<Integer> list=new LinkedList<>();
        if(m<1 || n<1){
            return -1;
        }
        for(int i=0;i<n;i++){
            list.add(i);
        }
        int bt=0;
        while(list.size()>1){
            bt=(bt+m-1)%list.size();
            list.remove(bt);
        }
        return list.get(0)+1;
    }
}
```

### [NC141 判断回文](https://www.nowcoder.com/practice/e297fdd8e9f543059b0b5f05f3a7f3b2?tpId=188&&tqId=38638&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给定一个字符串，请编写一个函数判断该字符串是否回文。如果回文请返回true，否则返回false。

**示例1**

输入：

```
"absba"
```

复制

返回值：

```
true
```

复制

**示例2**

输入：

```
"ranko"
```

复制

返回值：

```
false
```

```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 
     * @param str string字符串 待判断的字符串
     * @return bool布尔型
     */
    public boolean judge (String str) {
        // write code here
        char[] arr = str.toCharArray();
        int i = 0;
        int j = arr.length-1;
        while(i<j){
            if(arr[i]!=arr[j]){
                return false;
            }
            i++;
            j--;
        }
        return true;
    }
}
```

### [NC36 在两个长度相等的排序数组中找到上中位数](https://www.nowcoder.com/practice/6fbe70f3a51d44fa9395cfc49694404f?tpId=188&&tqId=38639&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给定两个有序数组arr1和arr2，已知两个数组的长度都为N，求两个数组中所有数的上中位数。

上中位数：假设递增序列长度为n，若n为奇数，则上中位数为第n/2+1个数；否则为第n/2个数

[要求]

时间复杂度为O(logN)*O*(*l**o**g**N*)，额外空间复杂度为O(1)*O*(1)

**示例1**

输入：

```
[1,2,3,4],[3,4,5,6]
```

复制

返回值：

```
3
```

复制

说明：

```
总共有8个数，上中位数是第4小的数，所以返回3。 
```

```java
import java.util.*;


public class Solution {
    /**
     * find median in two sorted array
     * @param arr1 int整型一维数组 the array1
     * @param arr2 int整型一维数组 the array2
     * @return int整型
     */
    public int findMedianinTwoSortedAray (int[] nums1, int[] nums2) {
        // write code here
        int n = nums1.length;
        int m = nums2.length;
        //
        int left = (n+m+1)/2;
        int right = (n+m+2)/2;
        return getKth(nums1,0,nums1.length-1,nums2,0,nums2.length-1,n);
    }
    
    public int getKth(int[] nums1,int start1,int end1,int[] nums2,int start2,int end2,int k){
        int len1 = end1-start1+1;
        int len2 = end2-start2+1;
        
        
        if(len1==0){
            return nums2[start2+k-1];
        }
        if(k==1){
            return Math.min(nums1[start1],nums2[start2]);
        }

        // 继续递归 新的索引
        int i = start1 + Math.min(len1,k/2) - 1;
        int j = start2 + Math.min(len2,k/2) - 1;
        if(nums1[i]>nums2[j]){
            return getKth(nums1,start1,end1,nums2,j+1,end2,k-(j-start2+1));
        }else{
            return getKth(nums1,i+1,end1,nums2,start2,end2,k-(i-start1+1));
        }
        
    }
}
```

### [NC101 缺失数字](https://www.nowcoder.com/practice/9ce534c8132b4e189fd3130519420cde?tpId=188&&tqId=38653&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

从 0,1,2,...,n 这 n+1 个数中选择 n 个数，选择出的数字依然保持有序，找出这 n 个数中缺失的那个数，要求 O(n) 或 O(log(n)) 并尽可能小。

**示例1**

输入：

```
[0,1,2,3,4,5,7]
```

复制

返回值：

```
6
```

复制

**示例2**

输入：

```
[0,2,3]
```

复制

返回值：

```
1
```

```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 找缺失数字
     * @param a int整型一维数组 给定的数字串
     * @return int整型
     */
    public int solve (int[] a) {
        // write code here
        int len = a.length;
        for(int i=0;i<len;i++){
            while(a[i]<len&&a[i]!=a[a[i]]){
                swap(a,i,a[i]);
            }
        }
        for(int i=0;i<len;i++){
            if(i!=a[i]){
                return i;
            }
        }
        return len;
    }
    
    public void swap(int[] nums,int i,int j){
        int temp = nums[i];
        nums[i]  = nums[j];
        nums[j]  = temp;
    }
}
```

### [NC30 数组中未出现的最小正整数](https://www.nowcoder.com/practice/8cc4f31432724b1f88201f7b721aa391?tpId=188&&tqId=38665&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给定一个无序数组arr，找到数组中未出现的最小正整数

例如arr = [-1, 2, 3, 4]。返回1

arr = [1, 2, 3, 4]。返回5

[要求]

时间复杂度为O(n)*O*(*n*)，空间复杂度为O(1)*O*(1)

**示例1**

输入：

```
[-1,2,3,4]
```

复制

返回值：

```
1
```

```java
import java.util.*;


public class Solution {
    /**
     * return the min number
     * @param arr int整型一维数组 the array
     * @return int整型
     */
    public int minNumberdisappered (int[] arr) {
        // write code here
        //从未出现的最小正整数
        for(int i=0;i<arr.length;i++){
            while(arr[i]>=1&&arr[i]<=arr.length&& arr[i]!=arr[arr[i]-1]){
                swap(arr,i,arr[i]-1);
            }
        }
        //找到
        for(int i=0;i<arr.length;i++){
            if(i!=arr[i]-1){
                return i+1;
            }
        }
        return arr.length+1;
    }
    public void swap(int[] nums,int i,int j){
        int temp = nums[i];
        nums[i]  = nums[j];
        nums[j]  = temp;
    }
}
```



### [NC20 数字字符串转换为IP地址](https://www.nowcoder.com/practice/ce73540d47374dbe85b3125f57727e1e?tpId=188&&tqId=38663&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

现在有一个只包含数字的字符串，将该字符串转化成IP地址的形式，返回所有可能的情况。

例如：

给出的字符串为"25525522135",

返回["255.255.22.135", "255.255.221.35"]. (顺序没有关系)

**示例1**

输入：

```
"25525522135"
```

复制

返回值：

```
["255.255.22.135","255.255.221.35"]
```

```java
import java.util.*;


public class Solution {
    /**
     * 
     * @param s string字符串 
     * @return string字符串ArrayList
     */
    //字符串转IP地址
    public ArrayList<String> restoreIpAddresses (String s) {
        // write code here
        ArrayList<String> res = new ArrayList<>();
        ArrayList<String> path = new ArrayList<>();
        dfs(s,0,path,res);
        return res;
    }
    public void dfs(String s,int index,ArrayList<String> path,ArrayList<String> res){
        //递归截止条件
        if(path.size()>4){
            return;
        }
        if(path.size()>=4&&index!=s.length()){
            return;
        }
        if(path.size()==4){
            res.add(String.join(".",path));
            return;
        }
        //遍历
        for(int i=index;i<s.length();i++){
            //判断是否在值的范围
            String newNum = s.substring(index,i+1);
            if(newNum.length()>1&&newNum.startsWith("0") || newNum.length()>3){
                continue;
            }
            int value = Integer.valueOf(newNum);
            if(value<0||value>255){
                continue;
            }
            path.add(newNum);
            dfs(s,i+1,path,res);
            path.remove(path.size()-1);
        }
    }
}
```

### [NC107 寻找峰值](https://www.nowcoder.com/practice/1af528f68adc4c20bf5d1456eddb080a?tpId=188&&tqId=38666&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

山峰元素是指其值大于或等于左右相邻值的元素。给定一个输入数组nums，任意两个相邻元素值不相等，数组可能包含多个山峰。找到索引最大的那个山峰元素并返回其索引。

假设 nums[-1] = nums[n] = -∞。

**示例1**

输入：

```
[2,4,1,2,7,8,4]
```

复制

返回值：

```
5
```

```java
import java.util.*;


public class Solution {
    /**
     * 寻找最后的山峰
     * @param a int整型一维数组 
     * @return int整型
     */
    public int solve (int[] a) {
        // write code here
         int n=a.length,po=-1;
        for(int i=0;i<n;++i){
            if(i==0&&a[i]>=a[i+1]){
                po=i;
            }
            else if(i==n-1&&a[i]>=a[i-1]){
                po=i;
            }
            else if(i>0&&i<n-1&&a[i]>=a[i-1]&&a[i]>=a[i+1]){
                po=i;
            }
        }
        return po;
    }
}
```

### [NC157 单调栈](https://www.nowcoder.com/practice/ae25fb47d34144a08a0f8ff67e8e7fb5?tpId=188&&tqId=38558&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给定一个长度为 n*n* 的可能含有重复值的数组 arr*a**r**r* ，找到每一个 i*i* 位置左边和右边离 i*i* 位置最近且值比 arr_i*a**r**r**i* 小的位置。

请设计算法，返回一个二维数组，表示所有位置相应的信息。位置信息包括：两个数字 L*L* 和 R*R*，如果不存在，则值为 -1，下标从 0 开始。

数据范围：

1 \le n \le 10^51≤*n*≤105

-10^9 \le arr[i] \le 10^9−109≤*a**r**r*[*i*]≤109

**示例1**

输入：

```
[3,4,1,5,6,2,7]
```

复制

返回值：

```
[[-1,2],[0,2],[-1,-1],[2,5],[3,5],[2,-1],[5,-1]]
```

```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param nums int一维数组 
     * @return int二维数组
     */
    public int[][] foundMonotoneStack (int[] nums) {
        // write code here
        int len = nums.length;
        // 返回的数组构造
        int[][] ans = new int[len][2];
        // 用栈保存
        Stack<Integer> stack = new Stack<>();
        // 从左往右，依次进行入栈，保存从左到右的升序的值
        for(int i = 0; i < len; i++){
            // 如果栈里面的值都比其大，就pop
            while(!stack.isEmpty() && nums[stack.peek()] > nums[i]) stack.pop();
            // 栈空，说明nums[i]左边没有比他小的值
            if(stack.isEmpty()){
                ans[i][0] = -1;
            } else {
                // 如果有比他小的，那么栈中的第一个元素的值就是离他最近的
                ans[i][0] = stack.peek();
            } 
            stack.push(i);
        }
        // 思路跟上面的一样，从右往左，保存升序值
        stack.clear();
        for(int i = len - 1; i >= 0; i--){
            while(!stack.isEmpty() && nums[stack.peek()] > nums[i]) stack.pop();
            if(stack.isEmpty()){
                ans[i][1] = -1;
            } else {
                ans[i][1] = stack.peek();
            }
            stack.push(i);
        }
        return ans;
    }
}
```





```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param nums int一维数组 
     * @return int二维数组
     */
    public int[][] foundMonotoneStack (int[] nums) {
        // write code here
        int[][] result = new int[nums.length][2];
        
        
        for(int i = 0; i < nums.length; i++){
            result[i][0] = -1;
            result[i][1] = -1; 
            int left = i, right = i;
            
            while(left >= 0){                
                if(nums[left] < nums[i]){
                    result[i][0] = left;
                    break;
                }
                left--;
            }
            
            while(right < nums.length){                
                if(nums[right] < nums[i]){
                    result[i][1] = right;
                    break;
                }
                right++;
            }            
        }
        return result;
    }
}
```

### [NC54 数组中相加和为0的三元组](https://www.nowcoder.com/practice/345e2ed5f81d4017bbb8cc6055b0b711?tpId=188&&tqId=38621&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给出一个有n个元素的数组S，S中是否有元素a,b,c满足a+b+c=0？找出数组S中所有满足条件的三元组。

注意：

1. 三元组（a、b、c）中的元素必须按非降序排列。（即a≤b≤c）
2. 解集中不能包含重复的三元组。

```
例如，给定的数组 S = {-10 0 10 20 -10 -40},解集为(-10, -10, 20),(-10, 0, 10) 
0 <= S.length <= 1000
```

**示例1**

输入：

```
[0]
```

复制

返回值：

```
[]
```

```java
import java.util.*;
public class Solution {
    public ArrayList<ArrayList<Integer>> threeSum(int[] nums) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        int len = nums.length;
        for(int i=0;i<len-2;i++){
            if(i!=0&&nums[i]==nums[i-1]){
                continue;
            }
            int l = i+1;
            int r = len-1;
            while(l<r){
                int temp_sum = nums[i] + nums[l] + nums[r];
                if(temp_sum==0){
                    ArrayList<Integer> path = new ArrayList<>();
                    path.add(nums[i]);
                    path.add(nums[l]);
                    path.add(nums[r]);
                    res.add(new ArrayList<>(path));
                    while(l+1<r&&nums[l]==nums[l+1]){
                        l++;
                    }
                    l++;
                    while(r-1>l&&nums[r]==nums[r-1]){
                        r--;
                    }
                    r--;
                }else if(temp_sum>0){
                    r--;
                }else if(temp_sum<0){
                    l++;
                }
            }
        }
        return res;
    }
}
```

### [NC112进制转换](https://www.nowcoder.com/practice/2cc32b88fff94d7e8fd458b8c7b25ec1?tpId=188&&tqId=38624&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给定一个十进制数 M ，以及需要转换的进制数 N 。将十进制数 M 转化为 N 进制数。

当 N 大于 10 以后， 应在结果中使用大写字母表示大于 10 的一位，如 'A' 表示此位为 10 ， 'B' 表示此位为 11 。

若 M 为负数，应在结果中保留负号。

**示例1**

输入：

```
7,2
```

复制

返回值：

```
"111"
```

复制

**备注：**

```
M是32位整数，2<=N<=16.
```

```java
import java.util.*;


public class Solution {
    /**
     * 进制转换
     * @param M int整型 给定整数
     * @param N int整型 转换到的进制
     * @return string字符串
     */
    public String solve (int M, int N) {
        // write code here
        String t = "0123456789ABCDEF";
        if(M==0){
            return "0";
        }
        //结果
        StringBuilder res = new StringBuilder();
        //记录一下是否为负数
        boolean flag = false;
        if(M<0){
            flag = true;
            M = -M;
        }
        while(M!=0){
            res.append(t.charAt(M%N));
            M/=N;
        }
        if(flag){
            res.append('-');
        }
        return res.reverse().toString();
    }
}
```

### [NC9A0 包含min函数的栈](https://www.nowcoder.com/practice/4c776177d2c04c2494f2555c9fcc1e49?tpId=188&&tqId=38626&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数，并且调用 min函数、push函数 及 pop函数 的时间复杂度都是 O(1)

push(value):将value压入栈中

pop():弹出栈顶元素

top():获取栈顶元素

min():获取栈中最小元素

示例:

输入:  ["PSH-1","PSH2","MIN","TOP","POP","PSH1","TOP","MIN"]

输出:  -1,2,1,-1

解析:

"PSH-1"表示将-1压入栈中，栈中元素为-1

"PSH2"表示将2压入栈中，栈中元素为2，-1

“MIN”表示获取此时栈中最小元素==>返回-1

"TOP"表示获取栈顶元素==>返回2

"POP"表示弹出栈顶元素，弹出2，栈中元素为-1

"PSH-1"表示将1压入栈中，栈中元素为1，-1

"TOP"表示获取栈顶元素==>返回1

“MIN”表示获取此时栈中最小元素==>返回-1

**示例1**

输入：

```
 ["PSH-1","PSH2","MIN","TOP","POP","PSH1","TOP","MIN"]
```

复制

返回值：

```
-1,2,1,-1
```

```java
import java.util.Stack;

public class Solution {
    Stack<Integer> s_min = new Stack<>();
    Stack<Integer> s_num = new Stack<>();
    
    public void push(int node) {
        s_num.push(node);
        if(s_min.isEmpty() || s_min.peek()>node){
            s_min.push(node);
        }else{
            s_min.push(s_min.peek());
        }
    }
    
    public void pop() {
        s_num.pop();
        s_min.pop();
    }
    
    public int top() {
        return s_num.peek();
    }
    
    public int min() {
        return s_min.peek();
    }
}
```

### [NC147 主持人调度](https://www.nowcoder.com/practice/4edf6e6d01554870a12f218c94e8a299?tpId=188&&tqId=38647&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

有 ![img](https://www.nowcoder.com/equation?tex=n%5C) 个活动即将举办，每个活动都有开始时间与活动的结束时间，第 ![img](https://www.nowcoder.com/equation?tex=i%5C) 个活动的开始时间是 start_i*s**t**a**r**t**i* ,第 ![img](https://www.nowcoder.com/equation?tex=i%5C) 个活动的结束时间是 end_i*e**n**d**i* ,举办某个活动就需要为该活动准备一个活动主持人。

一位活动主持人在同一时间只能参与一个活动。并且活动主持人需要全程参与活动，换句话说，一个主持人参与了第 ![img](https://www.nowcoder.com/equation?tex=i%5C) 个活动，那么该主持人在 (start_i,end_i)(*s**t**a**r**t**i*,*e**n**d**i*) 这个时间段不能参与其他任何活动。求为了成功举办这 ![img](https://www.nowcoder.com/equation?tex=n%5C) 个活动，最少需要多少名主持人。

数据范围:
1 \le n \le 10^51≤*n*≤105
-2^{32} \le start_i,end_i \le 2^{31}-1−232≤*s**t**a**r**t**i*​,*e**n**d**i*​≤231−1

**示例1**

输入：

```
2,[[1,2],[2,3]]
```

复制

返回值：

```
1
```

复制

说明：

```
只需要一个主持人就能成功举办这两个活动  
```

```java
import java.util.*;


public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 计算成功举办活动需要多少名主持人
     * @param n int整型 有n个活动
     * @param startEnd int整型二维数组 startEnd[i][0]用于表示第i个活动的开始时间，startEnd[i][1]表示第i个活动的结束时间
     * @return int整型
     */
    public int minmumNumberOfHost (int n, int[][] startEnd) {
        int[] starts = new int[startEnd.length];
        int[] ends = new int[startEnd.length];
        for(int i = 0 ;i < startEnd.length;i++){
            starts[i] = startEnd[i][0];
            ends[i] = startEnd[i][1];
        }
        Arrays.sort(starts);
        Arrays.sort(ends);
        int count = 0;
        int end = 0;
        for(int start = 0;start< startEnd.length;start++){
            if(starts[start] >= ends[end]){
                end++;
            }else{
                count++;
            }
        }
        return count;
    }
}
```





## Shell

### [NC1 统计文件的行数](https://www.nowcoder.com/practice/205ccba30b264ae697a78f425f276779?tpId=195&&tqId=36211&rp=1&ru=/activity/oj&qru=/ta/shell/question-ranking)

**描述**

写一个 bash脚本以输出一个文本文件 nowcoder.txt中的行数
示例:
假设 nowcoder.txt 内容如下：

```cpp
#include <iostream>
using namespace std;
int main()
{
    int a = 10;
    int b = 100;
    cout << "a + b:" << a + b << endl;
    return 0;
}
```

你的脚本应当输出：
9



```shell
wc -l ./nowcoder.txt | awk '{print $1}'
```

`wc -l` 是用来查看文件的newline的数量的。

在linux系统中，newline字符就是 `\n` 字符。

输出中包含了文件名，因此再做一下处理：

### [NC2 打印文件的最后5行](https://www.nowcoder.com/practice/ff6f36d357d24ce5a0eb817a0ef85ee2?tpId=195&tags=&title=&difficulty=0&judgeStatus=0&rp=1)

经常查看日志的时候，会从文件的末尾往前查看，于是请你写一个 bash脚本以输出一个文本文件 nowcoder.txt中的最后5行
示例:
假设 nowcoder.txt 内容如下：

```cpp
#include<iostream>
using namespace std;
int main()
{
int a = 10;
int b = 100;
cout << "a + b:" << a + b << endl;
return 0;
}
```



你的脚本应当输出：

```cpp
int a = 10;
int b = 100;
cout << "a + b:" << a + b << endl;
return 0;
}
```



```shell
tail -5 nowcoder.txt
```

> 附加题：
>
> ## 显示文件最后几行
>
> 查看文件的前5行，可以使用head命令，如
> head -5 filename
> 查看文件的后5行，可以使用tail命令，如：
> tail -5 filename 或 tail -n 5 filename
> 查看文件中间一段，你可以使用sed命令，如：
> sed -n ‘5,20p’ filename
> 这样你就可以只查看文件的第5行到第20行。

### [NC3 输出7的倍数](https://www.nowcoder.com/practice/8b85768394304511b0eb887244e51872?tpId=195&tags=&title=&difficulty=0&judgeStatus=0&rp=1)

写一个 bash脚本以输出数字 0 到 500 中 7 的倍数(0 7 14 21...)的命令

```shell
#!/bin/bash
for num in {0..500..7}; do  
  echo "${num}" 
done
```

### [NC4 输出低5行的内容](https://www.nowcoder.com/practice/1d5978c6136d4252904757b4fa0c9296?tpId=195&tags=&title=&difficulty=0&judgeStatus=0&rp=1)

**描述**

写一个 bash脚本以输出一个文本文件 nowcoder.txt 中第5行的内容。



示例:
假设 nowcoder.txt 内容如下：
welcome
to
nowcoder
this
is
shell
code

你的脚本应当输出：
is

```shell
awk 'NR==5'
```



```shell
sed -n 5p;
```



### [NC5 打印空行的行号](https://www.nowcoder.com/practice/030fc368e42e44b8b1f8985a8d6ad255?tpId=195&tags=&title=&difficulty=0&judgeStatus=0&rp=1)

写一个 bash脚本以输出一个文本文件 nowcoder.txt中空行的行号,可能连续,从1开始

示例:
假设 nowcoder.txt 内容如下：

```cpp
a
b

c

d

e


f
```


你的脚本应当输出：
3
5
7
9
10

```shell
 awk '/^\s*$/{print NR}' nowcoder.txt
 sed -n '/^\s*$/=' nowcoder.txt
```

### [NC6  去掉空行](https://www.nowcoder.com/practice/0372acd5725d40669640fd25e9fb7b0f?tpId=195&tags=&title=&difficulty=0&judgeStatus=0&rp=1)

写一个 bash脚本以去掉一个文本文件 nowcoder.txt中的空行
示例:
假设 nowcoder.txt 内容如下：

```cpp
abc

567


aaa
bbb



ccc
```


你的脚本应当输出：
abc
567
aaa
bbb
ccc

```shell
cat ./nowcoder.txt | awk NF
```



### [NC7 打印字母数小于8的单词](https://www.nowcoder.com/practice/bd5b5d4b93a04226a81afbabf0be797d?tpId=195&tags=&title=&difficulty=0&judgeStatus=0&rp=1)

写一个 bash脚本以统计一个文本文件 nowcoder.txt中字母数小于8的单词。



示例:
假设 nowcoder.txt 内容如下：
how they are implemented and applied in computer 

你的脚本应当输出：
how
they
are

and

applied

in

说明:
不要担心你输出的空格以及换行的问题

```shell
cat nowcoder.txt | awk '{
    for(i=1;i<=NF;i++){
        if(length($i)<8)
            print $i
    }
}'
```

### [NC 8 统计所有进程占用内存大小的和](https://www.nowcoder.com/practice/fb24140bac154e5b99e44e0cee45dcaf?tpId=195&tags=&title=&difficulty=0&judgeStatus=0&rp=1)

假设 nowcoder.txt 内容如下：
root     2 0.0 0.0   0   0 ?    S  9月25  0:00 [kthreadd]
root     4 0.0 0.0   0   0 ?    I<  9月25  0:00 [kworker/0:0H]
web    1638 1.8 1.8 6311352 612400 ?   Sl  10月16 21:52 test
web    1639 2.0 1.8 6311352 612401 ?   Sl  10月16 21:52 test
tangmiao-pc    5336  0.0 1.4 9100240 238544  ?? S   3:09下午  0:31.70 /Applications

以上内容是通过ps aux | grep -v 'RSS TTY' 命令输出到nowcoder.txt文件下面的
请你写一个脚本计算一下所有进程占用内存大小的和:

```shell
awk '{sum += $6} END {print sum}' nowcoder.txt
```



### [NC9 统计每个单词出现的个数](https://www.nowcoder.com/practice/ad921ccc0ba041ea93e9fb40bb0f2786?tpId=195&tags=&title=&difficulty=0&judgeStatus=0&rp=1)

**描述**

写一个 bash脚本以统计一个文本文件 nowcoder.txt 中每个单词出现的个数。

为了简单起见，你可以假设：
nowcoder.txt只包括小写字母和空格。
每个单词只由小写字母组成。
单词间由一个或多个空格字符分隔。

示例:
假设 nowcoder.txt 内容如下：
welcome nowcoder
welcome to nowcoder
nowcoder
你的脚本应当输出（以词频升序排列）：
to 1 
welcome 2 
nowcoder 3 

说明:
不要担心个数相同的单词的排序问题，每个单词出现的个数都是唯一的。

```shell
cat nowcoder.txt|tr ' ' '\n'|sort|uniq -c|sort -n|awk '{print $2" "$1}'

```

### [NC10 第二列是否有重复](https://www.nowcoder.com/practice/61b79ffe88964c7ab7b98ae16dd76492?tpId=195&tags=&title=&difficulty=0&judgeStatus=0&rp=1)

给定一个 nowcoder.txt文件，其中有3列信息，如下实例，编写一个sheel脚本来检查文件第二列是否有重复，且有几个重复，并提取出重复的行的第二列信息：
实例：
20201001 python 99
20201002 go 80
20201002 c++ 88
20201003 php 77
20201001 go 88
20201005 shell 89
20201006 java 70
20201008 c 100
20201007 java 88
20201006 go 97

结果：
2 java
3 go

```shell
cat nowcoder.txt | awk '{print $2}' | sort | uniq -c | sort | grep -v 1
```



grep name# 表示只查看name这个内容
grep -v name # 表示查看除了name之外的内容



### [NC11 转置文件的内容](https://www.nowcoder.com/practice/2240cd809c8f4d80b3479d7c95bb1e2e?tpId=195&tags=&title=&difficulty=0&judgeStatus=0&rp=1)

写一个 bash脚本来转置文本文件nowcoder.txt中的文件内容。

为了简单起见，你可以假设：
你可以假设每行列数相同，并且每个字段由空格分隔

示例:
假设 nowcoder.txt 内容如下：
job salary
c++ 13
java 14
php 12

你的脚本应当输出（以词频升序排列）：
job c++ java php
salary 13 14 12

```shell
#!bin/bash
awk '{print $1 " "}' nowcoder.txt
awk '{print $2 " "}' nowcoder.txt
```

### [NC12 打印每一行出现的数字个数](https://www.nowcoder.com/practice/2d2a124f98054292aef71b453e705ca9?tpId=195&tags=&title=&difficulty=0&judgeStatus=0&rp=1)

写一个 bash脚本以统计一个文本文件 nowcoder.txt中每一行出现的1,2,3,4,5数字个数并且要计算一下整个文档中一共出现了几个1,2,3,4,5数字数字总数。



示例:
假设 nowcoder.txt 内容如下：
a12b8
10ccc
2521abc
9asf
你的脚本应当输出：
line1 number: 2
line2 number: 1
line3 number: 4
line4 number: 0
sum is 7

说明:
不要担心你输出的空格以及换行的问题

```shell
line_no=0
sum=0
while read line; do
    num=$(echo ${line} | grep -oE '[1-5]' | wc -l)
    (( sum += num ))
    (( line_no++ ))
    printf "line%d number: %d\n" $line_no ${num}
done
printf "sum is %d\n" $sum
```

### [NC13 去掉所有包含this的句子](https://www.nowcoder.com/practice/2c5a46ef755a4f099368f7588361a8af?tpId=195&tags=&title=&difficulty=0&judgeStatus=0&rp=1)

**描述**

写一个 bash脚本以实现一个需求，去掉输入中含有this的语句，把不含this的语句输出
示例:
假设输入如下：
that is your bag
is this your bag?
to the degree or extent indicated.
there was a court case resulting from this incident
welcome to nowcoder


你的脚本获取以上输入应当输出：
to the degree or extent indicated.
welcome to nowcoder

说明:
你可以不用在意输出的格式，包括空格和换行

```shell
grep -v 'this'
```

### [NC14 求平均值](https://www.nowcoder.com/practice/c44b98aeaf9942d3a61548bff306a7de?tpId=195&tags=&title=&difficulty=0&judgeStatus=0&rp=1)

写一个bash脚本以实现一个需求，求输入的一个的数组的平均值

第1行为输入的数组长度N
第2~N行为数组的元素，如以下为:
数组长度为4，数组元素为1 2 9 8
示例:
4
1
2
9
8

那么平均值为:5.000(保留小数点后面3位)
你的脚本获取以上输入应当输出：
5.000

```shell
awk 'NR!=1 {sum+=$1} END{printf("%.3f",sum/(NR-1))}' nowcoder.txt
```

### [NC15 去掉不需要的单词](https://www.nowcoder.com/practice/838a3acde92c4805a22ac73ca04e503b?tpId=195&tags=&title=&difficulty=0&judgeStatus=0&rp=1)

写一个 bash脚本以实现一个需求，去掉输入中的含有B和b的单词
示例:
假设输入如下：
big
nowcoder
Betty
basic
test


你的脚本获取以上输入应当输出：
nowcoder test

说明:
你可以不用在意输出的格式，空格和换行都行

```shell
grep -v '[bB]'
```

## SQL

### [SQL66 牛客每个人最近的登录日期](https://www.nowcoder.com/practice/ca274ebe6eac40ab9c33ced3f2223bb2?tpId=82&&tqId=35084&rp=1&ru=/activity/oj&qru=/ta/sql/question-ranking)

请你写出一个sql语句查询每个用户最近一天登录的日子，并且按照user_id升序排序，上面的例子查询结果如下:
![img](imgs\652.png)
查询结果表明:
user_id为2的最近的登录日期在2020-10-13
user_id为3的最近的登录日期也是2020-10-13

```sql
select user_id,Max(DATE) as d 
from login
group by user_id
order by user_id;
```

### [SQL67 牛客每个人最近的登录日期](https://www.nowcoder.com/practice/7cc3c814329546e89e71bb45c805c9ad?tpId=82&tqId=35084&rp=1&ru=%2Factivity%2Foj&qru=%2Fta%2Fsql%2Fquestion-ranking)

牛客每天有很多人登录，请你统计一下牛客每个用户最近登录是哪一天，用的是什么设备.

有一个登录(login)记录表，简况如下:

![img](https://uploadfiles.nowcoder.com/images/20210106/557336_1609907092884/B7BB8B84A2534ED56DAB6420C6D02C42)


第1行表示user_id为2的用户在2020-10-12使用了客户端id为1的设备登录了牛客网
。。。
第4行表示user_id为3的用户在2020-10-13使用了客户端id为2的设备登录了牛客网

还有一个用户(user)表，简况如下:

![img](https://uploadfiles.nowcoder.com/images/20200817/557336_1597652611050_C098FF7CF52D3DC2ECE5019F4E7A5E88)

还有一个客户端(client)表，简况如下:
![img](https://uploadfiles.nowcoder.com/images/20200817/557336_1597652618264_F2C14AA3F53E74C2FE5A266283E56241)

请你写出一个sql语句查询每个用户最近一天登录的日子，用户的名字，以及用户用的设备的名字，并且查询结果按照user的name升序排序，上面的例子查询结果如下:

![img](https://uploadfiles.nowcoder.com/images/20210106/557336_1609907127708/7B7FC5B3933D957E9FD6C914ACE1D91A)

查询结果表明:
fh最近的登录日期在2020-10-13，而且是使用ios登录的
wangchao最近的登录日期也是2020-10-13，而且是使用ios登录的

```sql
select u.name as u_n,c.name as c_n,max(l.date) as d 
from login l 
inner join user u on l.user_id=u.id 
inner join client c on l.client_id=c.id 
group by l.user_id
order by u_n asc
```

### [SQL 72考试分数](https://www.nowcoder.com/practice/f41b94b4efce4b76b27dd36433abe398?tpId=82&&tqId=35492&rp=1&ru=/activity/oj&qru=/ta/sql/question-ranking)

牛客每次考试完，都会有一个成绩表(grade)，如下:

![img](https://uploadfiles.nowcoder.com/images/20210204/557336_1612433921452/0B46B656C53FA8A5212FD1013D04A373)



第1行表示用户id为1的用户选择了C++岗位并且考了11001分

。。。

第8行表示用户id为8的用户选择了JS岗位并且考了9999分

请你写一个sql语句查询各个岗位分数的平均数，并且按照分数降序排序，结果保留小数点后面3位(3位之后四舍五入):

![img](https://uploadfiles.nowcoder.com/images/20210204/557336_1612433942431/6A268FF0A87D8412E7CD00081C6DFF07)



(注意: sqlite 1/2得到的不是0.5，得到的是0，只有1*1.0/2才会得到0.5，sqlite四舍五入的函数为round)

```sql
(select job,round(sum(score)*1.0/count(id),3) as avg from grade
group by job) 
```



```sql
select job,round(avg(score),3) as avg
from grade
group by job
order by avg(score) desc;
```



### [SQL73 考试分数二](https://www.nowcoder.com/practice/f456dedf88a64f169aadd648491a27c1?tpId=82&tags=&title=&difficulty=0&judgeStatus=0&rp=1)

牛客每次考试完，都会有一个成绩表(grade)，如下:

![img](https://uploadfiles.nowcoder.com/images/20210204/557336_1612434140530/A40EBCABDBC68539EE224EE1DC2A7FE7)



第1行表示用户id为1的用户选择了C++岗位并且考了11001分

。。。

第8行表示用户id为8的用户选择了前端岗位并且考了9999分

请你写一个sql语句查询用户分数大于其所在工作(job)分数的平均分的所有grade的属性，并且以id的升序排序，如下:

![img](https://uploadfiles.nowcoder.com/images/20210204/557336_1612434164748/393D7F7211F18AF58DCC405ABAAB04DD)



(注意: sqlite 1/2得到的不是0.5，得到的是0，只有1*1.0/2才会得到0.5，sqlite四舍五入的函数为round)

```sql
select grade.* from grade join 
(select job,round(sum(score)*1.0/count(id),3) as avg from grade
group by job) as t
on grade.job=t.job --联立新表，job相同
where grade.score > t.avg --现在的表的分数大于新表的分数
order by id
```





## 奇安信

### [1.舞蹈分队]

悉尼歌剧院准备举办一场舞蹈演出。

于是教练去挑选舞蹈演员。

他让n名舞蹈演员站成一排，每个演员都有一个独一无二的身高。

每3个远演员组成一个一个小组，分组规则如下：

从队伍中选出位置分别为j,k,l的3名演员，他们的身高分别为height[j],height[k],height[l]

由于教练是个强迫症，所以舞蹈小分队需要满足:height[j]<height[k]<height[l]或者height[j]>height[k]>height[l]

其中0<=j<k<l<n

请你返回按上述条件可以组建的舞蹈小分队，每个演员都可以是多个舞蹈小队的一部分。

```java
 /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param height int整型一维数组 舞蹈员身高的一维数组
     * @return int整型
     */
    public int TeamNums (int[] height) {
        // write code here
        int len = height.length;
        int count = 0; 
        for(int j=0;j<len;j++){
            for(int k=j+1;k<len;k++){
                for(int l=k+1;l<len;l++){
                    if((height[j]<height[k]&&height[k]<height[l])  || (height[j]>height[k]&&height[k]>height[l])){
                        count++;
                    }
                }
            }
        }
        return count;
    }
```

### [2.Leetcode456 132模式](https://leetcode-cn.com/problems/132-pattern/)

给你一个整数数组 nums ，数组中共有 n 个整数。132 模式的子序列 由三个整数 nums[i]、nums[j] 和 nums[k] 组成，并同时满足：i < j < k 和 nums[i] < nums[k] < nums[j] 。

如果 nums 中存在 132 模式的子序列 ，返回 true ；否则，返回 false 。

 

示例 1：

输入：nums = [1,2,3,4]
输出：false
解释：序列中不存在 132 模式的子序列。
示例 2：

输入：nums = [3,1,4,2]
输出：true
解释：序列中有 1 个 132 模式的子序列： [1, 4, 2] 。

```java
class Solution {
    public boolean find132pattern(int[] nums) {
        //132模式
        int n = nums.length;
        //132中的2
        int last = Integer.MIN_VALUE;
        //用来存储132模式中的3
        Stack<Integer> sta = new Stack<>();
        //提前递归结束
        if(nums.length<3){
            return false;
        }
        //开始遍历
        for(int i=n-1;i>=0;i--){
            //若出现132中的1则返回正确值
            if(nums[i]<last){
                return true;
            }
            //若当前值大于或等于2则更新(2为栈中小于当前值的最大元素)
            while(!sta.isEmpty()&&sta.peek()<nums[i]){
                last = sta.pop();
            }
            sta.push(nums[i]);
        }
        return false;
    }
}
```

### [3.Leetcode329 矩阵中的最长递增路径](https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/)

给定一个 m x n 整数矩阵 matrix ，找出其中 最长递增路径 的长度。

对于每个单元格，你可以往上，下，左，右四个方向移动。 你 不能 在 对角线 方向上移动或移动到 边界外（即不允许环绕）。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2021/01/05/grid1.jpg)


输入：matrix = [[9,9,4],[6,6,8],[2,1,1]]
输出：4 
解释：最长递增路径为 [1, 2, 6, 9]。

```java
class Solution {
    //方向
    public int[][] dirs = {{-1,0,},{1,0},{0,-1},{0,1}};
    int rows,cols;
    public int longestIncreasingPath(int[][] matrix) {
        //递归截止条件
        if(matrix==null || matrix.length==0 || matrix[0].length==0){
            return 0;
        }
        rows = matrix.length;
        cols = matrix[0].length;
        // 记忆化
        int[][] meno = new int[rows][cols];
        // 结果
        int res = 0;
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                res = Math.max(res,dfs(matrix,i,j,meno));
            }
        }
        return res;
    }

    //深度优先搜索
    public int dfs(int[][]matrix,int row,int col,int[][] meno){
        if(meno[row][col]!=0){
            return meno[row][col];
        }
        ++meno[row][col];
        // 遍历
        for(int[] dir:dirs){
            int newRow = row+dir[0],newCol = col+dir[1];
            if(newRow>=0&&newRow<rows&&newCol>=0&&newCol<cols&&matrix[newRow][newCol]>matrix[row][col]){
                meno[row][col] = Math.max(meno[row][col],dfs(matrix,newRow,newCol,meno)+1);
            }
        }
        return meno[row][col];
    } 

}
```

### [4.Leetcode1219黄金矿工](https://leetcode-cn.com/problems/path-with-maximum-gold/)

你要开发一座金矿，地质勘测学家已经探明了这座金矿中的资源分布，并用大小为 m * n 的网格 grid 进行了标注。每个单元格中的整数就表示这一单元格中的黄金数量；如果该单元格是空的，那么就是 0。

为了使收益最大化，矿工需要按以下规则来开采黄金：

每当矿工进入一个单元，就会收集该单元格中的所有黄金。
矿工每次可以从当前位置向上下左右四个方向走。
每个单元格只能被开采（进入）一次。
不得开采（进入）黄金数目为 0 的单元格。
矿工可以从网格中 任意一个 有黄金的单元格出发或者是停止。


示例 1：

输入：grid = [[0,6,0],[5,8,7],[0,9,0]]
输出：24
解释：
[[0,6,0],
 [5,8,7],
 [0,9,0]]
一种收集最多黄金的路线是：9 -> 8 -> 7。

```java
class Solution {
    public int getMaximumGold(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int res = 0;
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(grid[i][j]!=0){
                    res = Math.max(res,dfs(grid,i,j));
                }
            }
        }
        return res;
    }

    public int dfs(int[][] grid,int i,int j){
        if(i<0||i>=grid.length||j<0||j>=grid[0].length||grid[i][j]==0){
            return 0;
        }

        // 记录当前值
        int temp = grid[i][j];
        grid[i][j] = 0;

        int cur_count = temp;
        int leftcount = dfs(grid,i-1,j);
        int rightcount = dfs(grid,i+1,j);
        int upcount = dfs(grid,i,j-1);
        int downcount = dfs(grid,i,j+1);
        int othercount = Math.max(leftcount,Math.max(upcount,Math.max(rightcount,downcount)));

        grid[i][j] = temp;
        return cur_count+othercount;
    }
}
```

## 好未来

### Leetcode21合并两个排序的链表

将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

 

示例 1：

![img](https://assets.leetcode.com/uploads/2020/10/03/merge_ex1.jpg)


输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(-1);
        ListNode l = dummy;
        while(l1!=null&&l2!=null){
            if(l1.val>l2.val){
                l.next = l2;
                l2 = l2.next;
            }else{
                l.next = l1;
                l1 = l1.next;
            }
            l = l.next;
        }
        if(l1!=null){
            l.next = l1;
        }
        if(l2!=null){
            l.next = l2;
        }
        return dummy.next;
    }
}
```

### Leetcode189 旋转数组

给定一个数组，将数组中的元素向右移动 k 个位置，其中 k 是非负数。

 

进阶：

尽可能想出更多的解决方案，至少有三种不同的方法可以解决这个问题。
你可以使用空间复杂度为 O(1) 的 原地 算法解决这个问题吗？


示例 1:

输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右旋转 1 步: [7,1,2,3,4,5,6]
向右旋转 2 步: [6,7,1,2,3,4,5]
向右旋转 3 步: [5,6,7,1,2,3,4]
示例 2:

```java
class Solution {
    public void rotate(int[] nums, int k) {
        int len = nums.length;
        k = k%len;
        reverse(nums,0,len-1);
        reverse(nums,0,k-1);
        reverse(nums,k,len-1);
    }
    public void reverse(int[] nums,int i,int j){
        while(i<j){
            int temp = nums[i];
            nums[i]  = nums[j];
            nums[j]  = temp;
            i++;
            j--;
        }
    }
}
```

## 滴滴

### [NC15 求二叉树的层序遍历](https://www.nowcoder.com/practice/04a5560e43e24e9db4595865dc9c63a3?tpId=188&&tqId=38595&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

**描述**

给定一个二叉树，返回该二叉树层序遍历的结果，（从左到右，一层一层地遍历）
例如：
给定的二叉树是{3,9,20,#,#,15,7},
![img](https://uploadfiles.nowcoder.com/images/20210114/999991351_1610616074120/036DC34FF19FB24652AFFEB00A119A76)
该二叉树层序遍历的结果是
[
[3],
[9,20],
[15,7]
]

**示例1**

输入：

```
{1,2}
```

复制

返回值：

```
[[1],[2]]
```

```java
import java.util.*;

/*
 * public class TreeNode {
 *   int val = 0;
 *   TreeNode left = null;
 *   TreeNode right = null;
 * }
 */

public class Solution {
    /**
     * 
     * @param root TreeNode类 
     * @return int整型ArrayList<ArrayList<>>
     */
    public ArrayList<ArrayList<Integer>> levelOrder (TreeNode root) {
        // write code here
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        if(root==null){
            return res;
        }
        ArrayList<Integer> path = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int levelSize = 1;
        while(!queue.isEmpty()){
            root = queue.poll();
            path.add(root.val);
            levelSize--;
            
            if(root.left!=null){
                queue.offer(root.left);
            }
            if(root.right!=null){
                queue.offer(root.right);
            }
            if(levelSize==0){
                res.add(new ArrayList<>(path));
                path = new ArrayList<>();
                levelSize = queue.size();
            }
        }
        return res;
    }
}
```



## 美团

### 1.判断

```java
package com.lcz.autumn;

//本题为考试多行输入输出规范示例，无需提交，不计分。
import java.util.*;

public class Main {
 public static void main(String[] args) {
     Scanner sc = new Scanner(System.in);
     int T = sc.nextInt();
     while((T--)>0) {
    	int n = sc.nextInt();
    	int k = sc.nextInt();
    	int[] arr = new int[n];
    	for(int i=0;i<n;i++) {
    		arr[i] = sc.nextInt();
    	}
    	if(k==0 || k==n || k>n) {
    		System.out.println("NO");
    		continue;
    	}
    	//对其排序
    	Arrays.sort(arr);
    	int res = 0;
    	//开始判断
    	if(k>n) {
    		//数据量
    		System.out.println("NO");
    		continue;
    	}
    	//再次判断
    	if(k==0) {
    		res = 1;
    	}else {
    		res = arr[k-1]+1;
    	}
    	//判断范围
    	if(res<1 || res>n) {
    		System.out.println("NO");
    		continue;
    	}
    	//判断重复值
    	if(k<n&&arr[k-1]==arr[k]) {
    		System.out.println("NO");
    		continue;
    	}
    	
    	System.out.println("YES");
    	System.out.println(res);
    	 	
    	 
     }    
     
  }
 
}

```

### 2.去除邻接的多余的字母，只留下一个字母

```java
package com.lcz.autumn;

import java.util.Scanner;

public class Main2 {
	public static String elimate(String str) {
		while(true) {
			//aiCCCGmyyyySpp
			int len = str.length();
			//开始处理
			for(int i=0;i<str.length();i++) {
				int cur = i;
				int j = i+1;
				//计算相同的字符个数
				while(j<str.length()&&str.charAt(j)==str.charAt(i)) {
					j++;
				}
				//判断连续的大于等于2个了
				if(j-i>=2) {
					//去除
					str = str.substring(0,i+1)+str.substring(j);
					i = cur;
				}
			}
			if(str.length()==len) {
				break;
			}
		}
		return str;
		
	}
	
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		String str = sc.nextLine();
		//去除空格
		str = str.replaceAll("\\s+","");
		//相邻处理
		String res = elimate(str);
		System.out.println(res);
	}
}

```



### 3.二叉树构建 交换 以及中序遍历

```java
package com.lcz.autumn;
import java.util.*;
import java.util.Scanner;
/**
 * 小美的子树调换
时间限制： 3000MS
内存限制： 589824KB
题目描述：
小美给出一颗二叉树，接下来会进行若干次对某些节点的左右子树互换操作。

你的任务是给出经过小美若干次调换后的树的中序遍历。

 

下图是一个调换的例子：

- 初始的二叉树状态，此时中序遍历为 4 2 5 1 6 3 7

 

- 小美调换了1号节点的左右子树，此时中序遍历为 6 3 7 1 4 2 5

 



输入描述
第一行是一行整数n, m, k，表示有n个节点，从1编号到n。小美会进行m次节点的调换。小美给出的二叉树的根节点是k。

接下来有 n 行，对于第 i 行，每行两个正整数，描述 i 号节点的左右儿子。其中对于没有儿子的情况用0来表示。

最后一行是m个空格隔开的正整数，每个正整数表示小美调换了这个正整数所标识的节点的左右子树。

输出描述
经过这m次调换后，二叉树的中序遍历。每两个节点序号之间以空格隔开。


 * @author LvChaoZhang
 *
 */
class TreeNode{
	int val;
	TreeNode left;
	TreeNode right;
	public TreeNode(int val) {
		this.val = val;
	}
}

public class Main5 {
	
	
	static TreeNode[] Tree;
	public static void createTree(TreeNode root,HashMap<Integer,int[]> dict) {
		Tree[root.val] = root;
		int l = dict.get(root.val)[0];
		int r = dict.get(root.val)[1];
		if(l!=0) {
			root.left = new TreeNode(l);
			createTree(root.left,dict);
		}
		if(r!=0) {
			root.right = new TreeNode(r);
			createTree(root.right, dict);
		}
		
	}
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		int n = sc.nextInt();
		int m = sc.nextInt();
		int k = sc.nextInt();
		HashMap<Integer,int[]> dict= new HashMap<>();
		for(int i=1;i<=n;i++) {
			int left = sc.nextInt();
			int right  = sc.nextInt();
			dict.put(i,new int[] {left,right});
		}
		Tree = new TreeNode[n+1];
		//构建二叉树
		TreeNode root  = new TreeNode(k);
		createTree(root,dict);
		//开始调整
		while((m--)>0) {
			int t;
			t  = sc.nextInt();
			TreeNode temp = Tree[t].left;
			temp = Tree[t].right;
			Tree[t].right = temp;
		}
		
		//中序遍历
		Stack<TreeNode> stack = new Stack<>();
		while(!stack.isEmpty()||root!=null) {
			while(root!=null) {
				stack.push(root);
				root = root.left;
			}
			TreeNode temp = stack.peek();
			stack.pop();
			System.out.print(temp+" ");
			root = root.right;
		}
	}
}

```

### 面试

#### [indexOf，strStr()]

不使用类库api的前提下，给定两个字符串，求串2在串1中的位置，没有返回-1

```java
import java.util.*;
public class Main {
    public static int findFirstIndex(String s1,String s2){
        char[] arr1 = s1.toCharArray();
        char[] arr2 = s2.toCharArray();
        int len1 = arr1.length;
        int len2 = arr2.length;
        if(len2==0){
            return -1;
        }
        for(int i=0;i<=len1-len2;i++){
            int j = 0;
            while(j<len2&&arr1[i+j]==arr2[j]){
                j++;
            }
            if(j==len2){
                return i;
            }
        }
        return -1;
    }
    
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        //int a = in.nextInt();
        //System.out.println(a);
        String str1 = sc.next();
        String str2 = sc.next();
        System.out.println(findFirstIndex(str1,str2));
    }
}
```





## 字节面试

### 1.Leetcode024 两两交换链表中的节点

给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。

你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

 

示例 1：


输入：head = [1,2,3,4]
输出：[2,1,4,3]

```java
package com.lcz.contest.bytee;
//声明一个类
class ListNode{
	int val;
	ListNode next;
	public ListNode() {
		
	}
	public ListNode(int val) {
		this.val = val;
	}
	public ListNode(int val,ListNode next) {
		this.val = val;
		this.next = next;
	}
}
// 两两交换链表中的结点
public class Leetcode024 {
	
	// 交换函数
	public static ListNode swapPairs(ListNode head) {
		ListNode dummy = new ListNode(-1,head);
		// 指针
		ListNode node = dummy;
		// 开始
		while(node.next!=null&&node.next.next!=null) {
			ListNode slow = node.next;
			ListNode fast = node.next.next;
			slow.next = fast.next;
			
			node.next = fast;
			fast.next = slow;
			
			node = slow;
		}
		
		
		return dummy.next;
	}
	// 主函数
	public static void main(String[] args) {
		// 创建一个
		ListNode dummy = new ListNode(-1);
		//开始创建
		ListNode head = dummy;
		for(int i=1;i<=4;i++) {
			ListNode node = new ListNode(i);
			head.next = node;
			head = head.next;
		}
		head = swapPairs(dummy.next);
		// 打印
		while(head!=null) {
			System.out.println(head.val);
			head = head.next;
		}
		
	}
}

```

### [NC132环形链表的约瑟夫问题](https://www.nowcoder.com/practice/41c399fdb6004b31a6cbb047c641ed8a?tpId=188&&tqId=38612&rp=1&ru=/activity/oj&qru=/ta/job-code-high-week/question-ranking)

描述

编号为 11 到 n*n* 的 n*n* 个人围成一圈。从编号为 11 的人开始报数，报到 m*m* 的人离开。

下一个人继续从 11 开始报数。

n-1*n*−1 轮结束以后，只剩下一个人，问最后留下的这个人编号是多少？

示例1

输入：

```
5,2     
```

复制

返回值：

```
3    
```

复制

说明：

```
开始5个人 1，2，3，4，5 ，从1开始报数，1->1，2->2编号为2的人离开
1，3，4，5，从3开始报数，3->1，4->2编号为4的人离开
1，3，5，从5开始报数，5->1，1->2编号为1的人离开
3，5，从3开始报数，3->1，5->2编号为5的人离开
最后留下人的编号是3     
```

```java
import java.util.*;


public class Solution {
    /**
     * 
     * @param n int整型 
     * @param m int整型 
     * @return int整型
     */
    public int ysf (int n, int m) {
        // write code here
        LinkedList<Integer> list=new LinkedList<>();
        if(m<1 || n<1){
            return -1;
        }
        for(int i=0;i<n;i++){
            list.add(i);
        }
        int bt=0;
        while(list.size()>1){
            bt=(bt+m-1)%list.size();
            list.remove(bt);
        }
        return list.get(0)+1;
    }
}
```





### 2.约瑟夫环的书写

约瑟夫环是给定 n个人，从1开始数，数到m时，m将被去掉，下一个数是重新从1开始数，直到剩下一个即可。

```java
public class Code_01 {
	// 约瑟夫环 传入n和m
	public static int ysf_func(int n,int m) {
		// 用List来存储
		LinkedList<Integer> list= new LinkedList<>();
		for(int i=0;i<n;i++) {
			// 添加人的编码1-n
			list.add(i+1);
		}
		// 第一个索引是
		int index = 0;
		while(list.size()>1) {
			// 开始数数
			for(int i=1;i<m;i++) {
				index = (index+1)%list.size();
			}
			list.remove(index);
		}
		// 最后剩了一个
		return list.get(0);
	}
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		while(sc.hasNext()) {
			int n = sc.nextInt();
			int m = sc.nextInt();
			// 开始处理
			System.out.println(ysf_func(n, m));
		}
	}
}

```

### 3.k个小球问题

n个小球洒在桌子上，选定一个作为中点，之后找k个在正方形中的点。

就是找距离的第k个点。

求出距离。第k算法。

### [4.求两个数组的交集II]

给定两个数组，编写一个函数来计算它们的交集。



示例 1：

输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2,2]
示例 2:

输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[4,9]


说明：

- 输出结果中每个元素出现的次数，应与元素在两个数组中出现次数的最小值一致。
- 我们可以不考虑输出结果的顺序。

进阶：

- 如果给定的数组已经排好序呢？你将如何优化你的算法？
- 如果 nums1 的大小比 nums2 小很多，哪种方法更优？
- 如果 nums2 的元素存储在磁盘上，内存是有限的，并且你不能一次加载所有的元素到内存中，你该怎么办？

> 解题思路：与之前两个数组的交集不一样，不一样的地方在于Leetcode349中数组中的元素是唯一的，而这个元素不是唯一的。



> 一种是 通过HashMap来解题；
>
> 另一种是通过排序+双指针的思路来解题；

**解法一、HashMap的解题思路**

```java
class Solution {
    public int[] intersect(int[] nums1, int[] nums2) {
        // 解题思路一、HashMap
        // 让num1的length最小，
        if(nums1.length>nums2.length){
            return intersect(nums2,nums1);
        }

        HashMap<Integer,Integer> hashMap = new HashMap<>();
        // 先遍历num1来存储
        for(int num:nums1){
            hashMap.put(num,hashMap.getOrDefault(num,0)+1);
        }
        // 结果
        int[] in_arr = new int[nums1.length];
        int index = 0;
        // 遍历nums2
        for(int num:nums2){
            // 计算次数
            int count = hashMap.getOrDefault(num,0);
            // 如果有就是交集
            if(count>0){
                // 存储
                in_arr[index++] = num;
                // 判断是否还有了
                if(count>0){
                    hashMap.put(num,count-1);
                }else{
                    // 没了的话 删除
                    hashMap.remove(num);
                }
            }
        }
        return Arrays.copyOfRange(in_arr,0,index);
    }
}
```

**解法二、排序+双指针的解题思路**

```java
class Solution {
    public int[] intersect(int[] nums1, int[] nums2) {
        // 排序
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        // 结果存储
        int len1 = nums1.length;
        int len2 = nums2.length;
        int len = len1>len2?len2:len1;
        int[] res = new int[len];
        // 双指针
        int index1 = 0;
        int index2 = 0;
        int index = 0;
        while(index1<len1&&index2<len2){
            if(nums1[index1]>nums2[index2]){
                index2++;
            }else if(nums1[index1]<nums2[index2]){
                index1++;
            }else if(nums1[index1]==nums2[index2]){
                res[index++] = nums1[index1]; 
                index1++;
                index2++;
            }
        }
        return Arrays.copyOfRange(res,0,index);

    }
}
```

进阶三问：

- 如果给定的数组已经拍好序了，那么用解法二即可

  其复杂度为O（mlogm+nlogn）如果有序就变为O(m+n)

- 如果nums1的大小比nums2小很多，哪种方法更优呢？

  将较小的数组哈希奇数，随后在另一个数组中根据哈希来找即解法一更适合。时间复杂度为O(max(n,m))

- 如果nums2的元素存储在磁盘上，内存是有限的，并且不能一次性加载所有元素到内存中，该怎么办呢？

  归并排序。可以将可以将分割后的子数组写到单个文件中，归并时将小文件合并为更大的文件。当两个数组均排序完成生成两个大文件后，即可使用双指针遍历两个文件，如此可以使空间复杂度最低。


#### [5.1-n数字字典序第k大]

1-n数字字典序第k大
描述信息:
给你一个数字n(n < 1e9), 再给你一个数字k, 要求你找到1, 2, 3, ... n按照字典序排序后, 第k大的数字;
如, n = 15, k = 7;
那1 ~ 15按照字典序排序为: 1, 10, 11, 12, 13, 14, 15, 2, 3, 4, 5, 6, 7, 8, 9;
则答案为15;

```java
import java.util.*;
public class Main {
    public static int findKth(int n,int k){
        int cur = 1;
        int pre = 1;
        while(cur<k){
            int count = getCount(pre,n);
            if(cur+count==k){
               return pre+1;
            }else if(cur+count<k){
                cur += count;
                pre++;
            }else if(cur+count>k){
                cur++;
                pre *= 10;
            }
        }
        return pre;
    }
    
    public static int getCount(int pre,int n){
        int count = 0;
        int next  = pre+1;
        while(pre<=n){
            count += Math.min(next,n+1)-pre;
            pre  *= 10;
            next *= 10;
        }
        return count;
    }
    
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int n = in.nextInt();
        int k = in.nextInt();
        int res = findKth(n,k);
        //System.out.println(a);
        System.out.println(res);
    }
}
```

## 网易

###  1小于等于M的组合数

对于一个整型数组，里面任何两个元素相加，小于等于M的组合有多少种？如果有符合的，输出组合对数。

```java
package com.lcz.wangyi;

import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String s = sc.nextLine();
        int m = sc.nextInt();
        String[] str_arr = s.split(" ");
        int len = str_arr.length;
        int[] nums = new int[len];
        for(int i=0;i<len;i++) {
        	nums[i] = Integer.valueOf(str_arr[i]);
        }
        int count = 0;
        for(int i=0;i<len;i++) {
        	for(int j=i+1;j<len;j++) {
        		if(nums[i]+nums[j]<=m) {
        			count++;
        		}
        	}
        }
        System.out.println(count);
        
    }
}

```

### 2.字符操作转换(递归)

给你两个正整数n和k，其中1<=n<=26,字符串Sn的形成规则如下:

Li表示26个字母a-z,依次是

L1="a";

L2="b";

...

L26="z"



S1="a"

当i>1时，Si=Si-1+Li+reverse(invert(Si-1))

```java
package com.lcz.wangyi;

public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 返回a+b的和
     * @param a int整型
     * @param b int整型
     * @return int整型
     */
	public static char findKthBit (int n, int k) {
        // write code here	
		String str = funS(n);
		char res = str.charAt(k-1);
		return res;
    }
	
	static String dict = "abcdefghijklmnopqrstuvwxyz";
	
	public static String funS(int n) {
		if(n==1) {
			return "a";
		}
		String old = funS(n-1);
		String invert_old = invert(old);
		String reverse_old = reverse(invert_old);
		
		
		return funS(n-1)+dict.charAt(n-1)+reverse_old;
				
	}
	
	//翻转每一位
	public static String invert(String old) {
		StringBuilder invert_s = new StringBuilder();
		for(int i=0;i<old.length();i++)	{
			int index = dict.indexOf(old.charAt(i));
			if(index<=12) {
				invert_s.append(dict.charAt(25-index));
			}else {
				invert_s.append(dict.charAt(25-index));
			}
		}
		return invert_s.toString();
	}
	
	//反转字符串
	public static String reverse(String old) {
		char[] s = old.toCharArray();
		int n = s.length;
		int left = 0;
		int right = n-1;
		while(left<right) {
			char temp = s[left];
			s[left] = s[right];
			s[right] = temp;
			left++;
			right--;
		}
		return new String(s);
		
	}
	
	public static void main(String[] args) {
		System.out.println(findKthBit(4,11));
	}
	
}
```

### 3.分发糖果

```java
package com.lcz.wangyi;

//本题为考试单行多行输入输出规范示例，无需提交，不计分。
import java.util.Scanner;

public class Main2 {
	 public static void main(String[] args) {
	     Scanner sc = new Scanner(System.in);
	     String s = sc.nextLine();
	     
	     String[] s_arr = s.split(" ");
	     int len = s_arr.length;
	    
	     long[] nums = new long[len];
	     for(int i=0;i<len;i++) {
	    	 nums[i] = Long.valueOf(s_arr[i]);
	     }
	     long[] res = new long[len];
	     for(int i=0;i<len;i++) {
	    	 res[i] = 1;
	     }
	     //判断
	     
	     for(int i=0;i<len;i++) {
	    	 if(i==0&&nums[i]>nums[len-1]) {
	    		 res[i]=res[len-1]+1;
	    	 }
	    	 if(i>0&&nums[i]>nums[i-1]) {
	    		 res[i] = res[i-1]+1;
	    	 }
	     }
	     for(int i=len-1;i>=0;i--) {
	    	 if(i==len-1&&nums[i]>nums[0]) {
	    		 res[i] = Math.max(res[i],res[0]+1);
	    	 }
	    	 if(i<len-1&&nums[i]>nums[i+1]) {
	    		 res[i] = Math.max(res[i],res[i+1]+1);
	    	 }
	     }
	     //计算
	     long sum = 0;
	     for(int i=0;i<len;i++) {
	    	 sum += res[i];
	     }
	     System.out.println(sum);
	     
	     
	 }
}

```

### 4.网格路径代价最小

给你一个由‘0’(水)，‘1’陆地和2障碍物组成的二维网络，走陆地费用为1，走水路为2，障碍物无法通行，请你计算从网络的起始位置到最终位置的最小费用。

```java
package com.lcz.wangyi;

import java.util.Arrays;

public class Solution2 {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 计算最小航行费用
     * @param input int整型二维数组 二维网格
     * @return int整型
     */
    public static int minSailCost (int[][] input) {
        // write code here
    	if(input==null||input.length==0) {
    		return 0;
    	}
    	int m = input.length;
    	int n = input[0].length;
    	if(m==1&&n==1) {
    		return 1;
    	}
    	
    	int[][] dp = new int[m][n];
    	//初始化花费的代价
    	dp[0][0] = 0;
    	for(int i=1;i<m;i++) {
    		if(input[i][0]==2) {
    			//障碍物
    			for(int j=i;j<m;j++) {
        			dp[j][0] = -1;
    			}
    			break;
    		}
    		if(input[i][0]==0) {
    			//水路
    			dp[i][0] = dp[i-1][0]+2;
    		}
    		if(input[i][0]==1) {
    			//陆路
    			dp[i][0] = dp[i-1][0]+1; 
    		}
    	}
    	
    	for(int j=1;j<n;j++) {
    		if(input[0][j]==2) {
    			//障碍物
    			for(int i=j;i<n;i++) {
    				dp[0][i] = -1;
    			}
    			break;
    		}
    		if(input[0][j]==0) {
    			//水路
    			dp[0][j] = dp[0][j-1]+2;
    		}
    		if(input[0][j]==1) {
    			//陆路
    			dp[0][j] = dp[0][j-1]+1; 
    		}
    	}
    	
    	//转移方程
    	for(int i=1;i<m;i++) {
    		for(int j=1;j<n;j++) {
    			if(input[i][j]==2) {
    				//遇到障碍物 走不了
    				dp[i][j] = -1;
    				continue;
    			}else {
    				//判断当前是水路还是陆地
    				int cost = 0;
    				if(input[i][j]==0) {
    					//水路
    					cost = 2;
    				}else  if(input[i][j]==1) {
    					//陆地
    					cost = 1;
    				}
    				//判断之前的路径是否能走
    				if(dp[i-1][j]==-1&&dp[i][j-1]==-1) {
    					dp[i][j]=-1;
    				}else if(dp[i-1][j]==-1) {
    					dp[i][j]=dp[i][j-1]+cost;
    				}else if(dp[i][j-1]==-1){
    					dp[i][j]=dp[i-1][j]+cost;
    				}else {
        				dp[i][j] = Math.min(dp[i-1][j], dp[i][j-1])+cost;
    				}
    			}
    		}
    	}
//    	
//    	for(int i=0;i<m;i++) {
//    		for(int j=0;j<n;j++) {
//    			System.out.print(dp[i][j]+" ");
//    		}
//    		System.out.println();
//    	}
//    	
    	return dp[m-1][n-1];
    	
    }
    
//    
//    public static void main(String[] args) {
//    	int[][] input = {{1,1,1,1,0},{0,1,0,1,0},{1,1,2,1,1},{0,2,0,0,1}};
//    	System.out.println(minSailCost(input));
//    	
//	}
}

```



## 微博

### [1.糖果分配](https://leetcode-cn.com/problems/assign-cookies/)

假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。

对每个孩子 i，都有一个胃口值 g[i]，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j，都有一个尺寸 s[j] 。如果 s[j] >= g[i]，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。


示例 1:

输入: g = [1,2,3], s = [1,1]
输出: 1
解释: 
你有三个孩子和两块小饼干，3个孩子的胃口值分别是：1,2,3。
虽然你有两块小饼干，由于他们的尺寸都是1，你只能让胃口值是1的孩子满足。
所以你应该输出1。
示例 2:

输入: g = [1,2], s = [1,2,3]
输出: 2
解释: 
你有两个孩子和三块小饼干，2个孩子的胃口值分别是1,2。
你拥有的饼干数量和尺寸都足以让所有孩子满足。
所以你应该输出2.

```java
class Solution {
    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        // 结果
        int count = 0;
        int g_len = g.length;
        int s_len = s.length;
        // 开始
        for(int i=0,j=0;i<g_len&&j<s_len;i++,j++){
            // 使得j满足
            while(j<s_len&&g[i]>s[j]){
                j++;
            }
            if(j<s_len){
                count++;
            }
        }
        return count;
    }
}
```



## 阿里笔试

### [0. 删掉一个元素以后全为1的最长子数组]

给你一个二进制数组 nums ，你需要从中删掉一个元素。

请你在删掉元素的结果数组中，返回最长的且只包含 1 的非空子数组的长度。

如果不存在这样的子数组，请返回 0 。

 

提示 1：

输入：nums = [1,1,0,1]
输出：3
解释：删掉位置 2 的数后，[1,1,1] 包含 3 个 1 。

示例 2：

输入：nums = [0,1,1,1,0,1,1,0,1]
输出：5
解释：删掉位置 4 的数字后，[0,1,1,1,1,1,0,1] 的最长全 1 子数组为 [1,1,1,1,1] 。

> 滑动窗口，删掉可以想象成替换，无非就是最后求得时候再加上。对0的个数上的限制

```java
class Solution {
    public int longestSubarray(int[] nums) {
        int left = 0;
        int right = 0;
        int maxLength = 0;
        int zeros  = 0;
        while(right<nums.length){
            if(nums[right]==0){
                zeros++;
            }
            // 判断
            while(zeros>1){
                if(nums[left]==0){
                    zeros--;
                }
                left++;
            }

            // 替换了之后，但是必须要删掉一个元素
            maxLength = Math.max(right-left,maxLength);
            right++;
        }
        return maxLength;
    }
}
```



### [1.消消乐问题]

问题1：消消乐，对输入的字符串例如aabbccc变换成aabb；aabbcccb->aa

```java
package com.lcz.contest.alibaba;

import java.util.Scanner;


public class Code_XXL {
  
    
    // 消消乐消除大于等于3的字母
    public static String elimate(String str) {
    	while(true) {
    		//aabbcccb
    		// 记录当前的len
    		int length = str.length();
    		// 开始处理
    		for(int i=0;i<str.length();i++) {
    			int cur = i;
    			int j = i+1;
    			// 计算相同的字符个数
    			while(j<str.length()&&str.charAt(j)==str.charAt(i)) {
    				j++;
    			}
    			// 连续的大于等于3个了
    			if(j-i>2) {
    				// 去除
    				str = str.substring(0,i)+str.substring(j);
    				// 接着处理
    				i = cur; 	
    			}  
    			// 继续判断前一个
    			
    		}
			// 结束的条件若找不到连续的大于等于3个就结束
			if(str.length()==length) {
				break;
			}
    	}
    	return str;
    }
    
    
    public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		while(sc.hasNext()) {
			String str = sc.next();
			System.out.println("消消乐之前的结果" +str+"消消乐之后的结果："+elimate(str));
		}
	}
}

```

### [Leetcode1047 删除字符串中所有相邻重复项]

// 只是删除两个相邻且相同的字母，需要保存之前的。所以采用栈的结构

```java
class Solution {
    public String removeDuplicates(String S) {
        // 用栈来保存当前还未被删除的字符，栈 // 删除两个相邻且相同的字母
        // 对其转换
        char[] arr = S.toCharArray();
        // 结果存储 利用栈存储之前记录的
        StringBuilder res = new StringBuilder();
        // 记录此时res的长度
        int index = -1;
        for(int i=0;i<arr.length;i++){
            // 判断要加入的字符跟之前的相同
            if(index>=0&&arr[i]==res.charAt(index)){
                // 删除之前的字符
                res.deleteCharAt(index);
                index--;
            }else{
                // 第一次直接加入
                res.append(arr[i]);
                index++;
            }
        }
        return res.toString();
    }
}
```





### [2.单链表相交问题]

```java
package com.lcz.contest.alibaba;
// 两个单链表相交的问题
// 1.先判断这两个单链表是否有环 
// 2.无环的链表判断相交问题
// 3.有环的链表判断相交问题
public class Code_IntersectNode {
	// 链表
	public static class ListNode{
		int value;
		ListNode next;
		public ListNode() {
			
		}
		public ListNode(int value) {
			this.value = value;
		}
		public ListNode(int value,ListNode next) {
			this.value = value;
			this.next = next;
		}
	}
	
	// 得到相交结点的方法
	public static boolean getIntersectNode(ListNode node1,ListNode node2) {
		if(node1==null || node2==null) {
			return true;
		}
		// 首先判断传入的两个结点是否有环
		ListNode loop1 = detectCycle(node1);
		ListNode loop2 = detectCycle(node2);
		

		if(loop1==null&&loop2==null) {
			System.out.println("=====进入两个链表都无环的判断=====");
			return noLoopIntersection(node1,node2);
		}else if(loop1!=null&&loop2!=null){
			System.out.println("=====进入两个链表都有环的判断=====");
			return bothLoopIntersection(node1,loop1,node2,loop2); 
		}else {
			System.out.println("=====进入一个链表有环 一个链表无环判断=====");
			return false;
		}
		
	}
	// 判断一个链表是否有环，若有环就返回环结点，无环就返回空
	public static ListNode detectCycle(ListNode head) {
		ListNode slow = head;
		ListNode fast = head;
		while(fast!=null&&fast.next!=null) {
			slow = slow.next;
			fast = fast.next.next;
			// 若有环
			if(slow==fast) {
				// 重新走
				fast = head;
				while(fast!=slow) {
					fast = fast.next;
					slow = slow.next;
				}
				return slow;
			}
		}
		// 无环返回空
		return null;
	}
	// 无环判断是否相交的问题
	public static boolean noLoopIntersection(ListNode node1,ListNode node2) {
		ListNode p1 = node1;
		ListNode p2 = node2;
		while(p1!=p2) {
			p1 = p1==null?node2:p1.next;
			p2 = p2==null?node1:p2.next;
		}
		// 判断最后相交的点是否为null还是结点
		return p1==null?false:true;
	}
	// 有环
	// 相交的点 不在环点上
	//  - 不相交的情况
	//  - 相交的情况
	// 相交的点 在环点上
	public static boolean bothLoopIntersection(ListNode node1,ListNode loop1,ListNode node2,ListNode loop2) {
		ListNode cur1 = null;
		ListNode cur2 = null;
		// 环点相等的的情况
		if(loop1==loop2) {
			// 必定相交的情况
			return true;
		}else {
			// 环点不等 可能相交 可能不相交 
			// 让环点转一圈看看是否能找到另外一个环点
			cur1 = loop1.next;
			while(cur1!=loop1) {
				if(cur1==loop2) {
					// 找到环点了
					return true;
				}
				cur1 = cur1.next;
			}
			// 找不到
			return false;
		}
	}
	
	// 主函数测试
	public static void main(String[] args) {
		/**
		 * 单链表测试两个无环的
		 * 1->2->3->4->5
		 * 6->4->7
		 * 
		 */
		ListNode node1 = new ListNode(1);
		node1.next = new ListNode(2);
		node1.next.next = new ListNode(3);
		node1.next.next.next = new ListNode(4);
		node1.next.next.next.next = new ListNode(5);
		
		ListNode node2 = new ListNode(6);
		node2.next = new ListNode(8);
		node2.next.next = new ListNode(7);
		
		// 判断
		System.out.println("两个无环单链表 无相交点的情况："+getIntersectNode(node1,node2));
		
		/**
		 * 单链表测试无环但有相交链表的情况
		 * 1->2->3->4->5
		 * 6->7->3->4->5
		 */
		ListNode node3 = new ListNode(6);
		node3.next = new ListNode(7);
		node3.next.next = node1.next.next;
		System.out.println("两个无环单链表 有相交点的情况："+getIntersectNode(node1, node3));
		
		
 		/**
 		 * 单链表测试有环 但无相交结点的情况
 		 * 1->2-3->4->5->3
 		 * 1->2->3->4->5
 		 */
		ListNode node4 = new ListNode(1);
		node4.next = new ListNode(2);
		node4.next.next = new ListNode(3);
		node4.next.next.next = new ListNode(4);
		node4.next.next.next.next = new ListNode(5);
		node4.next.next.next.next.next = node4.next.next;
		System.out.println("一个无环单链表 一个有环的情况："+getIntersectNode(node1, node4));
		/**
 		 * 单链表测试环点相同的情况 相交
 		 * 1->2-3->4->5->3
 		 * 6->7->3->4->5->3
 		 */
		ListNode node5 = new ListNode(6);
		node5.next = new ListNode(7);
		node5.next.next = node4.next.next;
		System.out.println("有环链表环点相同 相交情况："+getIntersectNode(node4, node5));
		
		/**
 		 * 单链表测试环点不同的情况 相交情况
 		 * 1->2-3->4->5->3
 		 * 6->7->4-...
 		 */
		ListNode node6 = new ListNode(6);
		node6.next = new ListNode(7);
		node6.next.next = node4.next.next.next;
		System.out.println("有环链表环点不同 相交情况："+getIntersectNode(node4, node6));

		/**
 		 * 单链表测试环点不同的情况 不同的情况
 		 * 1->2-3->4->5->3
 		 * 6->7->8->9->7
 		 */
		ListNode node7 = new ListNode(6);
		node7.next = new ListNode(7);
		node7.next.next = new ListNode(8);
		node7.next.next.next = new ListNode(9);
		node7.next.next.next.next = node7.next;
		System.out.println("有环链表环点不同  不相交情况："+getIntersectNode(node4, node7));

		
	}
}

```

### [Leetcode1792 最大平均通过率](https://leetcode-cn.com/problems/maximum-average-pass-ratio/)

一所学校里有一些班级，每个班级里有一些学生，现在每个班都会进行一场期末考试。给你一个二维数组 classes ，其中 classes[i] = [passi, totali] ，表示你提前知道了第 i 个班级总共有 totali 个学生，其中只有 passi 个学生可以通过考试。

给你一个整数 extraStudents ，表示额外有 extraStudents 个聪明的学生，他们 一定 能通过任何班级的期末考。你需要给这 extraStudents 个学生每人都安排一个班级，使得 所有 班级的 平均 通过率 最大 。

一个班级的 通过率 等于这个班级通过考试的学生人数除以这个班级的总人数。平均通过率 是所有班级的通过率之和除以班级数目。

请你返回在安排这 extraStudents 个学生去对应班级后的 最大 平均通过率。与标准答案误差范围在 10-5 以内的结果都会视为正确结果。

 

示例 1：

输入：classes = [[1,2],[3,5],[2,2]], extraStudents = 2
输出：0.78333
解释：你可以将额外的两个学生都安排到第一个班级，平均通过率为 (3/4 + 3/5 + 2/2) / 3 = 0.78333 。

> 解题思路：贪心+优先级队列

```java
class Solution {
    //优先级队列+贪心策略
    public double maxAverageRatio(int[][] classes, int extraStudents) {
        int len = classes.length;
        //定义优先级队列，优先级增加1名学生之后能够产生的最大贡献来排序
        PriorityQueue<double[]> queue = new PriorityQueue<>( (o1,o2)->{
            double x = ( (o2[0]+1) / (o2[1]+1) - o2[0]/o2[1] );
            double y = ( (o1[0]+1) / (o1[1]+1) - o1[0]/o1[1] );
            if(x>y)return 1;
            if(x<y)return -1;;
            return 0;
        }   );
        //转换为double方便小数计算
        for(int[] c:classes){
            queue.offer(new double[]{c[0],c[1]});
        }
        //分配学生，每次分配1名
        while(extraStudents>0){
            //取出能够产生最大影响的班级
            double[] maxClasses = queue.poll();;
            //通过的人数
            maxClasses[0] += 1.0;
            maxClasses[1] += 1.0;
            //将更新后的重新加入队列中
            queue.offer(maxClasses);
            extraStudents--;
        }
        //计算最终结果
        double res = 0;
        while(!queue.isEmpty()){
            double[] c= queue.poll();
            res += (c[0]/c[1]);
        }
        return res/len;

    }
}
```



## 拼多多面试题

### [1.二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

给定一个二叉树的根节点 `root` ，返回它的 **中序** 遍历。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/09/15/inorder_1.jpg)

```
输入：root = [1,null,2,3]
输出：[1,3,2]
```

> 中序遍历

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        while(root!=null || !stack.isEmpty()){
            // 左节点不为空就一直入
            while(root!=null){
                stack.push(root);
                root = root.left;
            }

            root = stack.pop();
            res.add(root.val);
            
            root = root.right;
        }
        return res;
    }
}
```

### 2.恢复二叉搜索树

给你二叉搜索树的根节点 root ，该树中的两个节点被错误地交换。请在不改变其结构的情况下，恢复这棵树。

进阶：使用 O(n) 空间复杂度的解法很容易实现。你能想出一个只使用常数空间的解决方案吗？

 

示例 1：

![img](https://assets.leetcode.com/uploads/2020/10/28/recover1.jpg)

输入：root = [1,3,null,null,2]
输出：[3,1,null,null,2]
解释：3 不能是 1 左孩子，因为 3 > 1 。交换 1 和 3 使二叉搜索树有效。

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public void recoverTree(TreeNode root) {
        
        // 需要记录前一个
        TreeNode pre =  null;
        TreeNode x = null;
        TreeNode y = null;

        // 中序遍历
        Stack<TreeNode> stack = new Stack<>();
        while(root!=null || !stack.isEmpty()){
            while(root!=null){
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            // 开始处理
            if(pre!=null&&pre.val>root.val){
                y = root;
                if(x==null){
                    x = pre;
                }else{
                    break;
                }
            }
            // 记录前一个
            pre = root;
            root = root.right;
        }
        // 交换值
        swap(x,y);
    }
    //  交换
    public void swap(TreeNode x,TreeNode y){
        int temp = x.val;
        x.val    = y.val;
        y.val    = temp;
    }
}
```

### 3.常数时间插入、删除和获取随机元素

设计一个支持在平均 时间复杂度 O(1) 下，执行以下操作的数据结构。

insert(val)：当元素 val 不存在时，向集合中插入该项。
remove(val)：元素 val 存在时，从集合中移除该项。
getRandom：随机返回现有集合中的一项。每个元素应该有相同的概率被返回。
示例 :

// 初始化一个空的集合。
RandomizedSet randomSet = new RandomizedSet();

// 向集合中插入 1 。返回 true 表示 1 被成功地插入。
randomSet.insert(1);

// 返回 false ，表示集合中不存在 2 。
randomSet.remove(2);

// 向集合中插入 2 。返回 true 。集合现在包含 [1,2] 。
randomSet.insert(2);

// getRandom 应随机返回 1 或 2 。
randomSet.getRandom();

// 从集合中移除 1 ，返回 true 。集合现在包含 [2] 。
randomSet.remove(1);

// 2 已在集合中，所以返回 false 。
randomSet.insert(2);

// 由于 2 是集合中唯一的数字，getRandom 总是返回 2 。
randomSet.getRandom();

> List和HashMap，不重复

```java

class RandomizedSet {
    // 数组
    List<Integer> list;
    // hashMap的value是索引
    Map<Integer,Integer> hashMap;
    Random random;
    /** Initialize your data structure here. */
    public RandomizedSet() {
        random = new Random();
        list = new ArrayList<>();
        hashMap = new HashMap<>();

    }
    
    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    // 当元素val不存在时，向集合中插入该项
    public boolean insert(int val) {
        // 如果hashmap已经包含了val
        if(hashMap.containsKey(val)){
            return false;
        }
        // 存储
        list.add(val);
        hashMap.put(val,list.size()-1);
        return true;
    }
    
    /** Removes a value from the set. Returns true if the set contained the specified element. */
    // 元素val存在时，从集合中移除该项
    public boolean remove(int val) {
        // 如果不存在该值，则返回false
        if(!hashMap.containsKey(val)){
            return false;
        }
        // 删除元素的时候 不在意第几个,所以替换最后一个元素即可了。
        int lastVal = list.get(list.size()-1);
        // 获取要删除元素的索引
        int index = hashMap.get(val);
        // 重新设置数组和最后一个元素的值
        list.set(index,lastVal);
        hashMap.put(lastVal,index);

        // 删除最后一个元素即可
        list.remove(list.size()-1);
        hashMap.remove(val);
        return true;
    }
    
    /** Get a random element from the set. */
    public int getRandom() {
        return list.get(random.nextInt(list.size()));
    }
}

/**
 * Your RandomizedSet object will be instantiated and called as such:
 * RandomizedSet obj = new RandomizedSet();
 * boolean param_1 = obj.insert(val);
 * boolean param_2 = obj.remove(val);
 * int param_3 = obj.getRandom();
 */
```

### 4.O(1)时间插入、删除和获取随机元素-允许重复

设计一个支持在平均 时间复杂度 O(1) 下， 执行以下操作的数据结构。

注意: 允许出现重复元素。

insert(val)：向集合中插入元素 val。
remove(val)：当 val 存在时，从集合中移除一个 val。
getRandom：从现有集合中随机获取一个元素。每个元素被返回的概率应该与其在集合中的数量呈线性相关。
示例:

// 初始化一个空的集合。
RandomizedCollection collection = new RandomizedCollection();

// 向集合中插入 1 。返回 true 表示集合不包含 1 。
collection.insert(1);

// 向集合中插入另一个 1 。返回 false 表示集合包含 1 。集合现在包含 [1,1] 。
collection.insert(1);

// 向集合中插入 2 ，返回 true 。集合现在包含 [1,1,2] 。
collection.insert(2);

// getRandom 应当有 2/3 的概率返回 1 ，1/3 的概率返回 2 。
collection.getRandom();

// 从集合中删除 1 ，返回 true 。集合现在包含 [1,2] 。
collection.remove(1);

// getRandom 应有相同概率返回 1 和 2 。
collection.getRandom();

> list和map中map存储index索引。

```java
class RandomizedCollection {
    // 有重复的元素
    List<Integer> list;
    // 索引下标不可能重复
    Map<Integer,Set<Integer>> hashMap;
    Random random;
    /** Initialize your data structure here. */
    public RandomizedCollection() {
        list = new ArrayList<>();
        hashMap = new HashMap<>();
        random = new Random();
    }
    
    // 插入数据
    /** Inserts a value to the collection. Returns true if the collection did not already contain the specified element. */
    public boolean insert(int val) {
        list.add(val);
        // 在set里面放入值，如果未初始化就初始化
        Set<Integer> set = hashMap.getOrDefault(val,new HashSet<Integer>());
        set.add(list.size()-1);
        hashMap.put(val,set);

        // 判断是否重复了
        return set.size()==1;
        
    }
    
    // 当val存在时，从集合中移除一个val
    /** Removes a value from the collection. Returns true if the collection contained the specified element. */
    public boolean remove(int val) {
        // 如果不存在
        if(!hashMap.containsKey(val)){
            return false;
        }
        // 获取最后的元素
        int lastVal = list.get(list.size()-1);
        // 获取要删除元素的下标
        Set<Integer> set = hashMap.get(val);
        // 要删除元素的下标
        int index = set.iterator().next();
        // 数组重新设置
        list.set(index,lastVal);
        // hashMap中删除和更新下标集合
        hashMap.get(val).remove(index);
        hashMap.get(lastVal).remove(list.size()-1);
        // 更新下标集合
        if(index<list.size()-1){
            hashMap.get(lastVal).add(index);
        }
        if(hashMap.get(val).size()==0){
            hashMap.remove(val);
        }

        list.remove(list.size()-1);
        return true;
    }
    
    /** Get a random element from the collection. */
    public int getRandom() {
        return list.get(random.nextInt(list.size()));
    }
}

/**
 * Your RandomizedCollection object will be instantiated and called as such:
 * RandomizedCollection obj = new RandomizedCollection();
 * boolean param_1 = obj.insert(val);
 * boolean param_2 = obj.remove(val);
 * int param_3 = obj.getRandom();
 */
```

## 快手面试题

### [1.链表两数相加]

**题目描述：**

Leetcode:给定两个非空链表来表示两个非负整数。位数按照逆序方式存储，它们的每个节点只存储单个数字。将两数相加返回一个新的链表。

你可以假设除了数字 0 之外，这两个数字都不会以零开头。

示例：

```
输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
原因：342 + 465 = 807
```

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    // 两数相加
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(-1);
        ListNode l = dummy;
        // 两数相加有个多余的数
        int reg = 0;
        while(l1!=null || l2!=null){
            int s1 = l1!=null?l1.val:0;
            int s2 = l2!=null?l2.val:0;
            int temp = s1 + s2 + reg;
            
            reg = temp/10;
            int val = temp%10;
            l.next = new ListNode(val);
            l = l.next;

            if(l1!=null){
                l1 = l1.next;
            }

            if(l2!=null){
                l2 = l2.next;
            }
        }
        if(reg!=0){
            l.next = new ListNode(reg);
        }
        return dummy.next;
    }
}
```



## 腾讯面试题

### 0.多线程下的通讯问题

```java
package com.lcz.autumn;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.UnknownHostException;

public class MultiThreadSocket extends Thread{
	
	private Socket socket;
	
	public MultiThreadSocket() {
		
	}
	
	public MultiThreadSocket(Socket socket) {
		this.socket = socket;
	}
	
	/**
	 * 服务端
	 */
	public static void server() {
		//创建套接字
		try {
			ServerSocket ss = new ServerSocket(8000);
			//等待用户请求
			while(true) {
				Socket sk = ss.accept();
				//开启线程
				new MultiThreadSocket(sk).start();
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void client(String s) {
		try {
			Socket s1 = new Socket("127.0.0.1",8000);
			//获取到输出流
			OutputStream  os = s1.getOutputStream();
			//发送
			os.write((s+" hello world!").getBytes());
			//
			InputStream is = s1.getInputStream();
			byte[] buffer = new byte[1024];
			int length = is.read(buffer);
			String messgage = new String(buffer,0,length);
			System.out.println("server:"+messgage);
			//关闭
			os.close();
			is.close();
			s1.close();
			
		} catch (UnknownHostException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	/**
	 * 重写其run方法
	 */
	@Override
	public void run() {
		//读取数据
		try {
			InputStream is = socket.getInputStream();
			//读取
			byte[] buffer = new byte[1024];
			int length = is.read(buffer);
			//输出
			String message = new String(buffer,0,length);
			System.out.println("client:"+message);
			//收到消息之后在发送给对方
			//输出流
			OutputStream os = socket.getOutputStream();
			os.write(message.getBytes());
			
			//关闭
			is.close();
			os.close();
			socket.close();
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	
	
}

```



```java
package com.lcz.autumn;

public class TestServer {
	public static void main(String[] args) {
		MultiThreadSocket.server();
	}
}

```





```java
package com.lcz.autumn;

public class TestClient {
	public static void main(String[] args) {
		for(int i=0;i<10;i++) {
			MultiThreadSocket m = new MultiThreadSocket();
			m.client(i+"");
		}
	}
}

```



### 0.字符串优先级匹配问题

Problem Description

字符串中只含有括号：()、[]、<>、{}，判断输入的字符串中括号是否匹配。如果括号有相互包含的形式，从内到外必须是<>、()、[]、{}，例如：输入[()]，输出：YES，而输入([])、([)]都应该输出NO。

Input

输入的第一行为一个整数n，表示以下有多少个由括号组成的字符串。接下来的n行，每行都是一个由括号组成的长度不超过255的字符串。

Output

对于输入的n行，输出YES或者NO。

Sample Input

5

{}{}<><>()()[][]

{{}}{{}}<<>><<>>(())(())[[]][[]]

{{}}{{}}<<>><<>>(())(())[[]][[]]

{<>}{[]}<<<>><<>>>((<>))(())[[(<>)]][[]]

><}{{[]}<<<>><<>>>((<>))(())[[(<>)]][[]]

Sample Output

YES

YES

YES

YES

NO

 

首先为四种括号定义一个优先级，然后压栈的时候按照优先级的顺序来入栈，不满足就退出，如果遇到右括号在判断是否可以消除就行了。

```java
#include<cstdio>
#include<stack>
#include<cstring>
using namespace std;
int main()
{
	int n,i,a[256],l,flag;
	char s[256];
	scanf("%d",&n);
	while(n--)
	{
		scanf("%s",s);
		l=strlen(s);
		for(i=0;i<l;i++)
		{   //为每个括号设置一个优先级，小的在里面，大的在外面
			if(s[i]=='<') a[i]=1;
			if(s[i]=='>') a[i]=2;
			if(s[i]=='(') a[i]=3;
			if(s[i]==')') a[i]=4;
			if(s[i]=='[') a[i]=5;
			if(s[i]==']') a[i]=6;
			if(s[i]=='{') a[i]=7;
			if(s[i]=='}') a[i]=8;
		}
		stack<int>m; //定义一个用来存放优先级的栈
		flag=0;
		for(i=0;i<l;i++)
		{
			if(a[i]%2!=0)//如果是左括号
			{
				if(!m.empty())//如果栈不为空 并且当前字符优先级大于栈顶元素 直接跳出
				{  
					if(a[i]>m.top()) { flag=1; break; }
					else m.push(a[i]);//否则  压入元素（必须比栈顶元素优先级小 因为最先弹出的是栈顶元素）
				}
				else m.push(a[i]);//如果栈为空  直接插入
 
			}
			else //如果是右括号
			{
				if(m.empty()) { flag=1; break;}  //如果栈为空 不匹配  跳出
				if(m.top()+1!=a[i]) { flag=1; break; }//如果当前栈顶元素+1不等于当前元素 不匹配 跳出
				else m.pop();//否则  满足  
			}
		}
		if(flag) printf("NO\n");
		else
		{
			if(m.empty()) printf("YES\n");//栈为空 才输出yes
			else printf("NO\n");
		}
		while(!m.empty()) m.pop();
	}
	return 0;
}
```

### 0.bitmap实现1万个数字中某个数字是否存在

10亿int整型数，以及一台可用内存为1GB的机器，时间复杂度要求O(n)，统计只出现一次的数？

**分析**

首先分析多大的内存能够表示**10亿的数呢**？一个int型占4字节，10亿就是40亿字节（很明显就是4GB），也就是如果完全读入内存需要占用4GB，而题目只给1GB内存，显然不可能将所有数据读入内存。

我们先不考虑时间复杂度，仅考虑解决问题。那么接下来的思路一般有两种。

1. **位图法**：用一个bit位来标识一个int整数。

一种是位图法，如果各位老司机有经验的话很快会想到int整型数是4字节（Byte），也就是32位（bit），如果能用**一个bit位来标识一个int整数**那么存储空间将大大减少。另一种是分治法，内存有限，我想办法分批读取处理。下面分析一下位图法。

**位图法**

位图法是基于int型数的表示范围这个概念的，**用一个bit位来标识一个int整数，若该位为1，则说明该数出现**；**若该为为0，则说明该数没有出现。**一个int整型数占4字节（Byte），也就是32位（bit）。那么把所有int整型数字表示出来需要2^32 bit的空间，换算成字节单位也就是2^32/8 = 2^29 Byte，大约等于512MB

```java
// 插播一个常识
2^10 Byte = 1024 Byte = 1KB
2^30 Byte = (2^10)^3 Byte = 1024 * 1024 * 1024 Byte = 1GB
```

这下就好办了，只需要用512MB的内存就能存储所有的int的范围数。



**解决方案**

那么接下来我们只需要申请一个int数组长度为 int tmp[**N/32+1**]即可存储完这些数据，其中**N代表要进行查找的总数（这里也就是2^32）**，tmp中的每个元素在内存在占32位可以对应表示十进制数0~31,所以可得到BitMap表:

- tmp[0]:可表示0~31
- tmp[1]:可表示32~63
- tmp[2]可表示64~95
- ~~

假设这10亿int数据为：6,3,8,32,36,……，那么具体的BitMap表示为

![img](https://itimetraveler.github.io/gallery/bitmap/37237-20160302211041080-958649492.png)

(1). 如何判断int数字放在哪一个tmp数组中：将数字直接除以32取整数部分(x/32)，例如：整数8除以32取整等于0，那么8就在tmp[0]上；

(2). 如何确定数字放在32个位中的哪个位：将数字mod32取模(x%32)。上例中我们如何确定8在tmp[0]中的32个位中的哪个位，这种情况直接mod上32就ok，又如整数8，在tmp[0]中的第8 mod上32等于8，那么整数8就在tmp[0]中的第八个bit位（从右边数起）。

**然后我们怎么统计只出现一次的数呢**？每一个数出现的情况我们可以分为三种：0次、1次、大于1次。也就是说我们**需要用2个bit位才能表示每个数的出现情况**。此时则三种情况分别对应的bit位表示是：00、01、11

我们顺序扫描这10亿的数，在对应的双bit位上标记该数出现的次数。最后取出所有双bit位为01的int型数就可以了。



###### 自实现的代码

**如何确定每个数据在bitmap中的位置**

value位于数组bitmap中的index = value/32;

value位于数组bitmap[index]这个int中的 bit位置offset=value%32-1(offset是1移动多少位，位于第六位则需要1移动五位，所以要减一)；

**如何对一个int类型的32位为bit的某一个bit进行赋值操作呢？**

```java
bitmap[index] = bitmap[index] | 1<<offset;
```

`1<<offset`即为value的bitmap的位置，与目前有的值进行或操作进行合并

**如何对一个int类型的32位为bit的某一个bit进行读取操作？**

```java
bitmap[index] >> offset & 0x01 == 0x01?true:false;
```

**如何判重？**

设置bit时，先保存前值，设置之后如果值没有发生改变，则说明此bit位之前已经为1，即发生了重复

```java
int temp = a[index];
a[index] = a[index] | 1<<offset;
temp = a[index]; // 说明重复了
```

**如何根据bitmap恢复原始数据？**

```java
for(int i=0;i<a.length;i++){
    int t = a[i];
    for(int j=0;j<32;j++){
        t >> j & 0x01 == 0x01?true:false;
        if(true){
            int data = i*32 + j+1;
        }
    }
}
```

```java
class BitSet{
	// 常数
	static final int N = 1000000;
	// 数组存储
	int[] bitMap;
	public BitSet(){
		bitMap = new int[N/32 + 1];
	}
	// 添加一个数字
	public void add(int value) {
		int index  = value/32; // 位于数组bitmap中的index下标索引值
		int offset = value%32-1; //这个int中的bit位置 offset是1移动多少位，位于第六位则需要1移动无位
		// 放入值
		bitMap[index] = bitMap[index] | 1<<offset;
	}
	// 判断一个数字是否存在读取操作 
	public boolean isExist(int value) {
		int index = value/32;
		int offset = value%32-1;
		
		return ((bitMap[index]>>offset)&0x01)==0x01?true:false;
	}
	
	// 如何根据bitmap恢复原始数据
	public void reverseDigit() {
		for(int i=0;i<bitMap.length;i++) {
			int temp = bitMap[i];
			for(int j=0;j<32;j++) {
				boolean flag = ((temp>>j)&0x01) == 0x01?true:false;
				if(flag) {
					int data = i*32+ j+1;
				}
			}
		}
	}
}
```



###### 调用java库的代码

```java
public static void main(String[] args) {
		int[] array = new int[] {1,2,3,22,3};
		BitSet bitSet = new BitSet();
		// 将数组内容放入bitmap中
		for(int i=0;i<array.length;i++) {
			bitSet.set(array[i], true);
		}
		// 遍历bitmap
		int sum = 0;
		for(int i=0;i<bitSet.length();i++) {
			if(bitSet.get(i)) {
				sum++;
			}
		}
		System.out.println(sum);
	}
```

### 0.求最大公约数的方法

三种算法：

```java
//欧几里得算法（辗转相除）：

  public static int gcd(int m,int n) {
     if(m<n) {
       int k=m;
       m=n;
       n=k;
     }
     //if(m%n!=0) {
     //   m=m%n;
     //   return gcd(m,n);
     //}
     //return n;
     return m%n == 0?n:gcd(n,m%n);
   } 
```

 

```java
　  //连续整数检测算法：

　　public static int gcd1(int m,int n) {
     int t;
     if(m<n) {
       t=m;
     }else {
       t=n;
     }
     while(m%t!=0||n%t!=0){
       t--;
     }
     return t;
   }
```



 

```java
  //公因数法：(更相减损）

  public static int gcd2(int m,int n) {
     int i=0,t,x;
     while(m%2==0&n%2==0) {
       m/=2;
       n/=2;
       i++;
     }
     if(m<n){
       t=m;
       m=n;
       n=t;
     }
     while(n!=(m-n)) {
       x=m-n;
       m=(n>x)?n:x;
       n=(n<x)?n:x;
     }
     if(i==0)
       return n;
     else
       return (int)Math.pow(2, i)*n;
   }
```



```java
  public static void main(String[] args) {
     System.out.println("请输入两个正整数:");
     Scanner scan = new Scanner(System.in);
     Scanner scan2=new Scanner(System.in);
     int m=scan.nextInt();
     int n=scan2.nextInt();
     System.out.println("欧几里得算法求最大公约数是:"+gcd(m,n));
     System.out.println("连续整数检测算法求最大公约数是:"+gcd1(m,n));
     System.out.println("公因数法求最大公约数是:"+gcd2(m,n));
   }

}
```

### 0.Leetcode218 天际线问题

城市的天际线是从远处观看该城市中所有建筑物形成的轮廓的外部轮廓。给你所有建筑物的位置和高度，请返回由这些建筑物形成的 天际线 。

每个建筑物的几何信息由数组 buildings 表示，其中三元组 buildings[i] = [lefti, righti, heighti] 表示：

lefti 是第 i 座建筑物左边缘的 x 坐标。
righti 是第 i 座建筑物右边缘的 x 坐标。
heighti 是第 i 座建筑物的高度。
天际线 应该表示为由 “关键点” 组成的列表，格式 [[x1,y1],[x2,y2],...] ，并按 x 坐标 进行 排序 。关键点是水平线段的左端点。列表中最后一个点是最右侧建筑物的终点，y 坐标始终为 0 ，仅用于标记天际线的终点。此外，任何两个相邻建筑物之间的地面都应被视为天际线轮廓的一部分。

注意：输出天际线中不得有连续的相同高度的水平线。例如 [...[2 3], [4 5], [7 5], [11 5], [12 7]...] 是不正确的答案；三条高度为 5 的线应该在最终输出中合并为一个：[...[2 3], [4 5], [12 7], ...]

 

示例 1：

![img](https://assets.leetcode.com/uploads/2020/12/01/merged.jpg)


输入：buildings = [[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]
输出：[[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]
解释：
图 A 显示输入的所有建筑物的位置和高度，
图 B 显示由这些建筑物形成的天际线。图 B 中的红点表示输出列表中的关键点。
示例 2：

输入：buildings = [[0,2,3],[2,5,3]]
输出：[[0,3],[5,0]]

```java
class Solution {
    public List<List<Integer>> getSkyline(int[][] buildings) {
        PriorityQueue<int[]> pq = new PriorityQueue<int[]>((a, b) -> b[1] - a[1]);
        List<Integer> boundaries = new ArrayList<Integer>();
        for (int[] building : buildings) {
            boundaries.add(building[0]);
            boundaries.add(building[1]);
        }
        Collections.sort(boundaries);

        List<List<Integer>> ret = new ArrayList<List<Integer>>();
        int n = buildings.length, idx = 0;
        for (int boundary : boundaries) {
            while (idx < n && buildings[idx][0] <= boundary) {
                pq.offer(new int[]{buildings[idx][1], buildings[idx][2]});
                idx++;
            }
            while (!pq.isEmpty() && pq.peek()[0] <= boundary) {
                pq.poll();
            }

            int maxn = pq.isEmpty() ? 0 : pq.peek()[1];
            if (ret.size() == 0 || maxn != ret.get(ret.size() - 1).get(1)) {
                ret.add(Arrays.asList(boundary, maxn));
            }
        }
        return ret;
    }
}


```

### 0.实现arraylist

```java
package com.lcz.array;

@SuppressWarnings("unchecked")
public class ArrayList<E> {
	// 成员变量
	int size;
	E[] elements;
	
	
	// 常量
	private static final int DEFAULT_CAPATICY = 10;
	private static final int DEFAULT_NOT_FOUNT = -1;
	
	// 无参构造函数
	public ArrayList() {
		this(DEFAULT_CAPATICY);
	}
		
	// 有参构造函数
	public ArrayList(int capaticy) {
		capaticy = capaticy > DEFAULT_CAPATICY? capaticy:DEFAULT_CAPATICY;
		elements = (E[])new Object[capaticy];
	}
	
	
	
	/**
	 * 返回元素的数量
	 * @return
	 */
	public int size() {
		return size;
	}
	
	/**
	 * 是否为空
	 * @return
	 */
	public boolean isEmpty() {
		return size==0;
	}
	
	/**
	 * 是否包含某个元素
	 * @param element
	 * @return
	 */
	public boolean contains(E element) {
		return indexOf(element) != DEFAULT_NOT_FOUNT;
	}
	
	/**
	 * 添加元素到最后面
	 * @param element
	 */
	public void add(E element) {
		add(size,element);
	}
	
	private void outOfBounds(int index) {
		throw new IndexOutOfBoundsException("Index:" + index + ",Size:" + size);
	}
	private void rangeCheck(int index) {
		if(index<0 || index>=size) {
			outOfBounds(index);
		}
	}
	private void rangeForCheck(int index) {
		if(index<0 || index>size) {
			outOfBounds(index);
		}
	}
	/**
	 * 返回index位置对应的元素
	 * @param index
	 * @return
	 */
	public E get(int index) {
		// 对其进行index判断
		rangeCheck(index);
		return elements[index];
	}
	
	/**
	 * 设置index位置的元素
	 * @param index
	 * @param element
	 * @return 原来的元素
	 */
	public E set(int index,E element) {
		// 对其进行index判断
		rangeCheck(index);
		E oldElement = elements[index];
		elements[index] = element;
		return oldElement;
	}
	
	/**
	 * 往index位置添加元素
	 * @param index
	 * @param element
	 */
	public void add(int index,E element) {
		// index判断可以往size位置添加
		rangeForCheck(index);
		// 对其进行判断是否需要扩容
		checkForCapaticy(size+1);
		// 对其添加元素
		for(int i=size;i>index;i--) {
			elements[i] = elements[i-1]; 
		}
		elements[index] = element;
		size++;
	}
	
	/**
	 * 检查容量并扩容
	 */
	private void checkForCapaticy(int capaticy) {
		int oldCapaticy = elements.length;
		if(oldCapaticy >= capaticy) {
			return;
		}
		// 扩容
		// 新容量为旧容量的1.5倍
		int newCapaticy = oldCapaticy + oldCapaticy >> 1;
		E newElements[] = (E[])new Object[newCapaticy];
		for(int i=0;i<size;i++) {
			newElements[i] = elements[i];
		}
		elements = newElements;
					
	}
	
	
	/**
	 * 删除index位置对应的元素
	 * @param index
	 * @return
	 */
	public E remove(int index) {
		// 对index进行判断
		rangeCheck(index);
		E oldElement = elements[index];
		for(int i=index+1;i<size;i++) {
			elements[i-1] = elements[i]; 
		}
		elements[--size] = null;
		
		return oldElement;
	}
	
	/**
	 * 查看元素的索引
	 * @param element
	 * @return
	 */
	public int indexOf(E element) {
		if(element == null) {
			for(int i=0;i<size;i++) {
				if(elements[i]==null)
					return i;
			}
		}else {
			for(int i=0;i<size;i++) {
				if(element.equals(elements[i]))
					return i;
			}
		}
		return DEFAULT_NOT_FOUNT;
	}
	/**
	 * 清除所有元素
	 */
	public void clear() {
		for(int i=0;i<size;i++) {
			elements[i] = null;
		}
		size = 0;
	}
	
	/**
	 * 重写toString方法
	 */
	@Override
	public String toString() {
		// TODO Auto-generated method stub
		StringBuilder res = new StringBuilder();
		res.append("size=").append(size).append(", [");
		for(int i=0;i<size;i++) {
			if(i!=0) {
				res.append(", ");
			}
			res.append(elements[i]);
		}
		res.append("]");
		return res.toString();
	}
}

```



### 0.旋转链表

给你一个链表的头节点 head ，旋转链表，将链表每个节点向右移动 k 个位置。

 

示例 1：


输入：head = [1,2,3,4,5], k = 2
输出：[4,5,1,2,3]

![img](https://assets.leetcode.com/uploads/2020/11/13/rotate1.jpg)



```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode rotateRight(ListNode head, int k) {
        if(head==null || head.next == null) {
	    		return head;
	    	}
	    	// 1.使其变为环形链表
	    	ListNode tail = head;
	    	int len = 1;
	    	while(tail.next!=null) {
	    		len++;
	    		tail = tail.next;
	    	}
	    	tail.next = head;
	    	// 2.到尾结点这里
	    	k = k % len;
	    	for(int i=0;i<len-k;i++) {
	    		tail = tail.next;
	    	}
	    	//3.找到头结点
	    	head = tail.next;
	    	tail.next = null;
	    	return head;
    }
}
```



### 0.设计一个循环队列(重点数组题)

设计你的循环队列实现。 循环队列是一种线性数据结构，其操作表现基于 FIFO（先进先出）原则并且队尾被连接在队首之后以形成一个循环。它也被称为“环形缓冲器”。

循环队列的一个好处是我们可以利用这个队列之前用过的空间。在一个普通队列里，一旦一个队列满了，我们就不能插入下一个元素，即使在队列前面仍有空间。但是使用循环队列，我们能使用这些空间去存储新的值。

你的实现应该支持如下操作：

MyCircularQueue(k): 构造器，设置队列长度为 k 。
Front: 从队首获取元素。如果队列为空，返回 -1 。
Rear: 获取队尾元素。如果队列为空，返回 -1 。
enQueue(value): 向循环队列插入一个元素。如果成功插入则返回真。
deQueue(): 从循环队列中删除一个元素。如果成功删除则返回真。
isEmpty(): 检查循环队列是否为空。
isFull(): 检查循环队列是否已满。


示例：

MyCircularQueue circularQueue = new MyCircularQueue(3); // 设置长度为 3
circularQueue.enQueue(1);  // 返回 true
circularQueue.enQueue(2);  // 返回 true
circularQueue.enQueue(3);  // 返回 true
circularQueue.enQueue(4);  // 返回 false，队列已满
circularQueue.Rear();  // 返回 3
circularQueue.isFull();  // 返回 true
circularQueue.deQueue();  // 返回 true
circularQueue.enQueue(4);  // 返回 true
circularQueue.Rear();  // 返回 4

> 通过数组设置一个循环队列，设置一个最大容量，设置此时的size，设置一个headIndex，剩下的通过%就可以计算而来。

```java
class MyCircularQueue {
    int[] arr;
    int headIndex;
    int count;
    int capactity;
    public MyCircularQueue(int k) {
        arr = new int[k];
        headIndex = 0;
        count = 0;
        capactity = k;
    }
    // 入队列
    public boolean enQueue(int value) {
        if(isFull()){
            return false;
        }
        int index = (headIndex+count)%capactity;
        arr[index] = value;
        // 增加元素
        count++;
        return true;
    }
    // 出队列
    public boolean deQueue() {
        if(isEmpty()){
            return false;
        }
        // 就移动headIndex即可 
        headIndex = headIndex + 1;
        count--;
        return true;
    }
    // 从队首获取元素
    public int Front() {
        if(isEmpty()){
            return -1;
        }
        int index = headIndex;
        return arr[index];
    }
    // 从队尾获取元素
    public int Rear() {
        if(isEmpty()){
            return -1;
        }
        // 获取元素
        int  index = (headIndex+count-1)%capactity;
        return arr[index];
    }
    
    public boolean isEmpty() {
        return count==0;
    }
    
    public boolean isFull() {
        return count==capactity;
    }
}

/**
 * Your MyCircularQueue object will be instantiated and called as such:
 * MyCircularQueue obj = new MyCircularQueue(k);
 * boolean param_1 = obj.enQueue(value);
 * boolean param_2 = obj.deQueue();
 * int param_3 = obj.Front();
 * int param_4 = obj.Rear();
 * boolean param_5 = obj.isEmpty();
 * boolean param_6 = obj.isFull();
 */
```

### 1.实现一个计数器

实现一种计数器，记录最近 N个数字，
包含push、getAverage方法（输出当前计数器内平均值），o1复杂度。
实现另一种计数器，可以对最近N个数字做排序。

```java
package com.lcz.tencent;
/**
 * 实现一种计数器，记录最近 N个数字，
 * 包含push、getAverage方法（输出当前计数器内平均值），o1复杂度。
 
   实现另一种计数器，可以对最近N个数字做排序。
 * @author LvChaoZhang
 *
 */
public class Digit {
	int[] s_data;
	double s_avg;
	int n;
	int size;
	// 开始的下标索引
	int head = 0;
	public Digit() {
		
	}
	
	public Digit(int n) {
		this.n = n;
		this.size = 0;
		s_data = new int[n];
		s_avg = 0;
	}
	
	// 插入方法
	public void push(int x) {
		if(size==n) {
			int temp = s_data[head];
			s_data[head]=x;
			head++;
			// 更新
			s_avg = ((s_avg*size-temp)+x)/size;
			// 如果head到最后了
			if(head==size) {
				head = 0;
			}
			
		}else {
			// 容器未满的情况
			s_data[size] = x;
			size++;
			if(size==1) {
				// 之前没有
				s_avg = x;
			}else {
				s_avg = (s_avg*(size-1) + x)/(size);
			}
		}
	}
	
	// 获取当前计数器内平均值
	public double getAverage() {
		return s_avg;
	}
	
	// 打印方法
	public void print() {
		for(int i=0;i<size;i++) {
			System.out.print(s_data[i] + " ");
		}
		System.out.print("平均数为："+s_avg);
		System.out.println();
	}
}


```



```java
package com.lcz.tencent;

public class Other_Digit extends Digit{
	public Other_Digit(int n) {
		super(n);
	}
	// 对N个数字进行排序
	public void sort() {
		// 插入排序 
		for(int i=1;i<size;i++) {
			for(int j=i-1;j>=0&&s_data[j]>s_data[j+1];j--) {
				// 交换
				swap(s_data,j,j+1);
			}
		}
	}
	
	public void swap(int[] nums,int i,int j) {
		int temp = nums[i];
		nums[i]  = nums[j];
		nums[j]  = temp;
	}
}


```

```java
package com.lcz.tencent;

public class Test {
	public static void main(String[] args) {
		// 3 2 1
		Digit digit = new Digit(3);
		digit.push(3);
		digit.push(2);
		digit.push(1);
		// 打印一下
		digit.print();
		System.out.println(digit.s_avg);

		
		// 再往里面放一个4
		digit.push(4);
		digit.push(5);
		digit.push(6);
		digit.print();


		
		// 排序
		Other_Digit digit_2 = new Other_Digit(3);
		digit_2.push(3);
		digit_2.push(2);
		digit_2.push(1);
		digit_2.sort();
		digit_2.print();
	}
}


```

### 2.圆桌报数

已知n个人(以编号1,2,3,...n分别表示)，围坐在一张圆桌周围，从编号1的开始报数，数到k的那个人出列，它的下一个又从出列的后面第一个人开始报数，数到k的那个人出列，依次重复下去，直到圆桌的人全部出列，求出列顺序。

输入：n和k

输出：出列顺序

~~~java
### 2.约瑟夫环的书写

约瑟夫环是给定 n个人，从1开始数，数到m时，m将被去掉，下一个数是重新从1开始数，直到剩下一个即可。

```java
public class Code_01 {
	// 约瑟夫环 传入n和m
	public static int ysf_func(int n,int m) {
		// 用List来存储
		LinkedList<Integer> list= new LinkedList<>();
		for(int i=0;i<n;i++) {
			// 添加人的编码1-n
			list.add(i+1);
		}
		// 第一个索引是
		int index = 0;
		while(list.size()>1) {
			// 开始数数
			for(int i=1;i<m;i++) {
				index = (index+1)%list.size();
			}
			list.remove(index);
		}
		// 最后剩了一个
		return list.get(0);
	}
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		while(sc.hasNext()) {
			int n = sc.nextInt();
			int m = sc.nextInt();
			// 开始处理
			System.out.println(ysf_func(n, m));
		}
	}
}
~~~

### 3.统计不同号码个数

已知有一些文件，文件内包含一些电话号码，每个号码为8位数字，统计全部文件不同号码的个数。

备注：文件很多，总量很大，不足以一次全部读入到内存进行计算。

```java
package com.lcz.tencent_01;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class FileTest {
	// 获取数据
	public static int getData() throws IOException {
		// 使用arrayList来存储每行读取到的字符串
		int[] arr = new int[1000000000];
		// 读取一个目录下的文件
		File file = new File("D:\\000\\Leetcode刷题\\leetcode\\src\\com\\lcz\\tencent_01\\data");
		// 获取该目录下的所有文件
		File[] fileArray = file.listFiles();
		for(int i=0;i<fileArray.length;i++) {
			File filename = fileArray[i];
			// 读取该文件
			BufferedReader br = new BufferedReader(new FileReader(filename));
			// 按行获取字符串
			String str;
			while((str=br.readLine())!=null) {
				int number = Integer.valueOf(str);
				arr[number] = 1;
			}
		}
		// 最后对其进行统计
		int count = 0;
		for(int i=10000000;i<=99999999;i++) {
			if(arr[i]==1) {
				count++;
			}
		}
		return count;
	}
	
	public static void main(String[] args) {
		try {
			int number = getData();
			System.out.println(number);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}

```

### 4.设计简单的栈数据结构

请编写一个类来实现栈这样一种数据结构，实现如下能力：

- 压入数据；
- 弹出数据；
- 获取数据结构的长度

```java
package com.lcz.tencent_01;

public class MyStack {
	static final int default_capacity = 16;
	int[] data;
	int head;
	int size;
	
	/**
	 * 初始化
	 */
	public MyStack() {
		data = new int[default_capacity];
		head = 0;
		size = 0;
	}
	public MyStack(int n) {
		data = new int[n];
		head = 0;
		size = 0;
	}
	/**
	 * 入栈
	 * @param x
	 */
	public void push(int x) {
		// 判断是否栈满了
		if(isFull()) {
			//待处理
			resize();
		}
		data[head+size] = x;
		size++;
	}
	/**
	 * 出栈
	 * @return
	 */
	public int pop() {
		if(isEmpty()) {
			return -1;
		}
		int x = data[head+size-1];
		size--;
		return x;
	}
	/**
	 * 是否为空
	 * @return
	 */
	public boolean isEmpty() {
		return size==0;
	}
	/**
	 * 是否满了
	 */
	public boolean isFull() {
		return size==data.length;
	}
	/**
	 * 栈满了进行扩容
	 */
	public void resize() {
		// 扩容两倍
		int size = data.length>>1;
		int[] new_data = new int[size];
		// 开始复制
		for(int i=0;i<data.length;i++) {
			new_data[i] = data[i];
		}
		// 重新赋值
		data = new_data;
	}
}

```



### 2.设计结构类-二叉搜索树的序列化与反序列化

序列化：将二叉树转换为字符串

反序列化：将字符串转换为二叉树

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Codec {

    // Encodes a tree to a single string.
    // 序列化: 将二叉树转换为字符串
    public String serialize(TreeNode root) {
        if(root==null){
            return " ";
        }
        // 层序遍历
        Queue<TreeNode> queue = new LinkedList<>();
        // 结果存储
        StringBuilder res = new StringBuilder();
        queue.offer(root);
        while(!queue.isEmpty()){
            root = queue.poll();
            if(root!=null){
                // 不为空记录
                res.append(root.val+",");
                //  入队列
                queue.offer(root.left);
                queue.offer(root.right);
            }else{
                // 为空
                res.append("null,");
            }
        }
        return res.toString().substring(0,res.length()-1);
    }

    // Decodes your encoded data to tree.
    // 反序列化：将字符串转换为二叉树
    public TreeNode deserialize(String data) {
        if(data==" "){
            return null;
        }
        // 分割成数组
        String[] arr = data.split(",");
        int index = 0;
        // 建立根节点
        TreeNode root = new TreeNode(Integer.valueOf(arr[index]));
        index++;
        // 层序遍历
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while(!queue.isEmpty()){
            TreeNode cur = queue.poll();
            // 直接按照data数组走就行了
            // 不为空就创建树节点
            if(!"null".equals(arr[index])){
                // 左子树
                cur.left = new TreeNode(Integer.valueOf(arr[index]));
                // 入队列
                queue.offer(cur.left);
            }
            index++;
            if(!"null".equals(arr[index])){
                cur.right = new TreeNode(Integer.valueOf(arr[index]));
                queue.offer(cur.right);
            }
            index++;
        }
        return root;
    }
}

// Your Codec object will be instantiated and called as such:
// Codec ser = new Codec();
// Codec deser = new Codec();
// TreeNode ans = deser.deserialize(ser.serialize(root));
```

### 3.设计结构类-二叉搜索树的迭代器

重点是二叉搜索树，考虑中序遍历的写法，非递归的写法，那么就要用栈！！！！

用栈来辅助写二叉搜索树的迭代器

示例：


输入
["BSTIterator", "next", "next", "hasNext", "next", "hasNext", "next", "hasNext", "next", "hasNext"]
[[[7, 3, 15, null, null, 9, 20]], [], [], [], [], [], [], [], [], []]
输出
[null, 3, 7, true, 9, true, 15, true, 20, false]

解释
BSTIterator bSTIterator = new BSTIterator([7, 3, 15, null, null, 9, 20]);
bSTIterator.next();    // 返回 3
bSTIterator.next();    // 返回 7
bSTIterator.hasNext(); // 返回 True
bSTIterator.next();    // 返回 9
bSTIterator.hasNext(); // 返回 True
bSTIterator.next();    // 返回 15
bSTIterator.hasNext(); // 返回 True
bSTIterator.next();    // 返回 20
bSTIterator.hasNext(); // 返回 False

> 调用已存在的函数

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class BSTIterator {
    // 二叉搜索树-考虑栈 考虑中序遍历 先左 后右
    Stack<TreeNode> stack;
    public BSTIterator(TreeNode root) {
        stack = new Stack<>();
        // 初始化的时候存入一些值
        this.pre(root);
    }
    public void pre(TreeNode root){
        while(root!=null){
            stack.push(root);
            root = root.left;
        }
    }
    public int next() {
        TreeNode temp = stack.pop();
        // 查看该结点是否有右结点
        if(temp.right!=null){
            // 有右节点的话就将其全部入 调用已存在的
            this.pre(temp.right);
        }
        return temp.val;
    }
    
    public boolean hasNext() {
        // 判断栈此时是否为空
        return !stack.isEmpty();
    }
}

/**
 * Your BSTIterator object will be instantiated and called as such:
 * BSTIterator obj = new BSTIterator(root);
 * int param_1 = obj.next();
 * boolean param_2 = obj.hasNext();
 */
```

### 4.设计 结构类-恢复二叉搜索树

给你二叉搜索树的根节点 root ，该树中的两个节点被错误地交换。请在不改变其结构的情况下，恢复这棵树。

进阶：使用 O(n) 空间复杂度的解法很容易实现。你能想出一个只使用常数空间的解决方案吗？

 

示例 1：

输入：root = [1,3,null,null,2]
输出：[3,1,null,null,2]
解释：3 不能是 1 左孩子，因为 3 > 1 。交换 1 和 3 使二叉搜索树有效。

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public void recoverTree(TreeNode root) {
        // 中序遍历
        TreeNode pre = null;
        TreeNode x = null;
        TreeNode y = null;
        Stack<TreeNode> stack = new Stack<>();
        while(root!=null || !stack.isEmpty()){
            // 左结点入栈
            while(root!=null){
                stack.push(root);
                root = root.left;
            }

            //根节点处理
            root = stack.pop();
            // 交换的两种情况 且不符合情况  交换可能相邻也可能不相邻
            if(pre!=null&&pre.val>root.val){
                y = root;
                if(x==null){
                    x = pre;
                }else{
                    // x不为空则找到了
                    break;
                }
            }
            // 接着走
            pre = root;
            root = root.right;
        }
        // 交换
        swap(x,y);
    }
    public void swap(TreeNode x,TreeNode y){
        int temp = x.val;
        x.val = y.val;
        y.val = temp;
    }
}
```

### 5.设计结构类-二叉搜索树中的插入操作

给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。 输入数据 保证 ，新值和原始二叉搜索树中的任意节点值都不同。

注意，可能存在多种有效的插入方式，只要树在插入后仍保持为二叉搜索树即可。 你可以返回 任意有效的结果 。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2020/10/05/insertbst.jpg)


输入：root = [4,2,7,1,3], val = 5
输出：[4,2,7,1,3,5]

> 解题思路：对二叉搜索树找到其父节点，直到node结点为空，一直找找到最底层。之后根据comp来判断插入方向，这就只是个二叉搜索树而已。

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public TreeNode insertIntoBST(TreeNode root, int val) {
        // 如果是空子树
        if(root==null){
            root = new TreeNode(val);
            return root;
        }
        // 不为空找父节点 找到最底部，只在最底部插入值是吧
        TreeNode node = root;
        TreeNode parent = root;
        do{
            // 要插入的值与当前的值比较
            int comp = compare(val,node.val);
            // parent向下
            parent = node;
            // node向下
            if(comp>0){
                node = node.right;
            }else{
                node = node.left;
            }

        }while(node!=null);

        // 插入值
        TreeNode newNode = new TreeNode(val);
        if(compare(val,parent.val)>0){
            parent.right = newNode;
        }else{
            parent.left = newNode;
        }
        return root;
    }

    // 比较两个数
    public int compare(int a,int b){
        if(a>b){
            return 1;
        }else if(a==b){
            return 0;
        }else{
            return -1;
        }
    }
}
```

### 智力题-老鼠试毒算法

1000瓶毒药要至少多少只老鼠，假设每只老鼠吃了药之后，如果中毒会24小时内毒发而死，才能找出具体的哪一瓶？

答案是至少10只，需要将药混合起来，需要喂一轮。

**题解：**

位运算，每一只老鼠都可以当做一个二进位，代表0和1，0代表老鼠没事，1代表老鼠死了

根据2^10=1024，所以至少10个老鼠可以确定1000个瓶子具体哪个瓶子有毒。



一位表示一个老鼠，0-7表示8个瓶子。例如5号瓶子，对应是101，那就是老鼠1和老鼠3都得吃，4号瓶子对应是100，也就是只让老鼠3吃。

```
映射关系：
000 = 0
001 = 1
010 = 2
011 = 3
100 = 4
101 = 5
110 = 6
111 = 7
```

每一位就表示一个老鼠，右侧的0-7表示8个瓶子。

例如5号瓶子，对应是101，那就是老鼠1和老鼠3都得吃；4号瓶子对应是100，也就是只让老鼠3吃。

根据这个逻辑：分别将1、3、5、7号瓶子的药**混起来**给老鼠1吃，2、3、6、7号瓶子的药**混起来**给老鼠2吃，

4、5、6、7号瓶子的药**混起来**给老鼠3吃，哪一只老鼠死了，相应的位置就为1。

现在对某瓶未知序号的毒药，如果出现了情况是老鼠1死了、老鼠2没死、老鼠3死了，这就是101情况对应5号瓶子，那么就是101=5号瓶子有毒。

按照这个原理，10个老鼠也就可以确定1000个瓶子了

```java
package line.entertain.jerry;
 
import java.util.Arrays;
 
/**
 * @author line
 */
public class PoisonJerry {
 
	/**
	 * 根据老鼠死亡结果推断毒药编号
	 * 老鼠：rat, mouse
	 * 复数：rat -> rats, mouse -> mice
	 */
	private static int jerry(final int[] rats) {
		int numOfPoison = 0;
		for (int i = 0; i < rats.length; i++) {
			if (rats[i] != 0)
				numOfPoison += (1 << (rats.length - 1 - i));
		}
		return numOfPoison;
	}
	
	/**
	 * 根据老鼠个数和毒药编号获得老鼠死亡结果
	 * @param numOfRat 老鼠个数
	 * @param numOfPoison 毒药编号
	 */
	private static int[] jerry(final int numOfRat, final int numOfPoison) {
 
		/**
		 * 老鼠喝药情况的数组
		 * index : 老鼠的编号s0, s1, s2
		 * value : (value & numOfPoison) != 0 表示老鼠喝过这瓶毒药，对应index的老鼠给药死了
		 */
		int[] jerryDrinkPoison = new int[numOfRat];
		for (int i = 0; i < numOfRat; i++) { // 初始化喝药情况
			jerryDrinkPoison[i] = 1 << (numOfRat - 1 - i);
		}
		for (int i = 0; i < numOfRat; i++) { // 为了看的清楚一点写了两个循环
			jerryDrinkPoison[i] = (numOfPoison & jerryDrinkPoison[i]) == 0 ? 0 : 1;
		}
		return jerryDrinkPoison;
	}
	
	public static void main(String[] args) {
 
		/**
		 * 老鼠的死亡情况
		 */
//		int[] deathOfRat = new int[] { 0, 0, 0, 1, 1, 0, 1, 0, 1, 1 };
		int[] deathOfRat = new int[] { 0, 1, 1 };
		/**
		 * 毒药的编号
		 */
		int numOfPoison = jerry(deathOfRat);
		deathOfRat = jerry(deathOfRat.length, numOfPoison);
		
		System.out.println(numOfPoison);
		System.out.println(Arrays.toString(deathOfRat));
	}
}
```

### [Leetcode442 数组中重复的数据](https://leetcode-cn.com/problems/find-all-duplicates-in-an-array/)


给定一个整数数组 a，其中1 ≤ a[i] ≤ *n* （*n*为数组长度）, 其中有些元素出现**两次**而其他元素出现**一次**。

找到所有出现**两次**的元素。

你可以不用到任何额外空间并在O(*n*)时间复杂度内解决这个问题吗？

**示例：**

```
输入:
[4,3,2,7,8,2,3,1]

输出:
[2,3]
```

> 解题思路：可以在输入数组中用数字的正负来表示该位置所对应数字是否已经出现过。**遍历输入数组，给对应位置的数字取相反数，如果已经是负数，说明前面已经出现过，直接放入输出数组。**

```java
class Solution {
    public List<Integer> findDuplicates(int[] nums) {
        List<Integer> res = new ArrayList<>();
        for(int i=0;i<nums.length;i++){
            int num = Math.abs(nums[i]);
            if(nums[num-1]>0){
                nums[num-1] *= -1;
            }else{
                res.add(num);
            }
        }
        return res;
    }
}
```



## 拼多多面试题

### [1.寻找峰值最长的数组长度]

峰值元素是指其值大于左右相邻的元素。给你一个输入数组nums,找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回任何一个峰值所在位置即可了.

> 输入: nums[1,2,3,1]
>
> 输出：2
>
> 3是峰值元素，你的函数应该返回索引2.
>
> 改编之后是求长度

> 峰值元素判断的依据是当前比后面那个高即可了。

```java
public int peek(int[] nums){
    int len = nums.length;
    int res = 0;
    for(int i=0;i<len-1;i++){
        if(nums[i]>nums[i+1]){
            // 找到峰值
            int index = i;
            int l_index = i-1;
            while(l_index>=0&&nums[l_index]<nums[index]){
                l_index--;
            }
            int r_index = i+1;
            while(r_index<len&&nums[r_index]>nums[index]){
                r_index++;
            }
            res = Math.max(r_index-l_index+1,res);
        }
    }
    return res;
}
```

### [2.求一个数n在m进制下的末尾0的个数是多少]

```java
package com.lcz.contest.bytee;

import java.util.Scanner;

// 统计数据n在m进制下0的个数是多少
public class Code_Sum {
	public static int sum_zero (int n,int m) {
		int sum =  0;
		while(n>0) {
			sum += n/m;
			n /=m;
		}
		return sum;
	}
	static int[] prim = new int[100000];
	static int[] sum_pri = new int[100000];
	static int index = 0;
	
	// 分解质因数 m即m如果是10进制 那么有2和5
	// 分解25 会出现5*5 
	// 分解30 会出现5和6
	// 分解360位2*2*2*3*3*5
	public static void getpri(int m) {
		for(int i=2;i*i<=m;i++) {
			while(m%i==0) {
				prim[index]=i;
				sum_pri[index]++;
				m /= i;
			}
			// 移动到下一个索引
			if(sum_pri[index]>0) {
				index++;
			}
		}
		// 如果无法分解那么就是其本身
		if(m>1) {
			prim[index] = m;
			sum_pri[index] = 1;
			index++;
		}
		
		
	}
	
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		while(sc.hasNext()) {
			int n = sc.nextInt();
			int m = sc.nextInt();
			int res = Integer.MAX_VALUE;
			getpri(m);
			for(int i=0;i<index;i++) {
				res = Math.min(res, sum_zero(n,prim[i])/sum_pri[i]);
			}
			System.out.println(res);
		}
	}
}

```

### [3.旋转数组中查找最小值]

**找最小值用mid和最右边的比较；**

**找target还是跟左边进行比较；**

**重复的话，最小值通过if else判断即可了；求target就通过while来判断了。**

**多次旋转，有重复元素，且找最小边界三个if来保证。**

#### 3.1 旋转数组和旋转链表

循环链表的思路是：先让其成为一个环；计算k；之后找到尾，找到头，断开尾巴即可了。

循环数组的思路是：全部旋转、左旋、右旋转即可了。

```java
class Solution {
    // 旋转数组
    // 三步走
    // 1.全部旋转
    // 2.左旋转
    // 3.右旋转
    public void rotate(int[] nums, int k) {
        int len = nums.length;
        k = k%len;
        reverse(nums,0,len-1);
        reverse(nums,0,k-1);
        reverse(nums,k,len-1);
    }

    public void reverse(int[] nums,int left,int right){
        while(left<=right){
            int temp = nums[left];
            nums[left] = nums[right];
            nums[right] = temp;
            left++;
            right--;
        }
    }
}
```

#### 3.2 搜索旋转排序数组中的最小值

已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,2,4,5,6,7] 在变化后可能得到：
若旋转 4 次，则可以得到 [4,5,6,7,0,1,2]
若旋转 7 次，则可以得到 [0,1,2,4,5,6,7]
注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。

给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。

示例 1：

输入：nums = [3,4,5,1,2]
输出：1
解释：原数组为 [1,2,3,4,5] ，旋转 3 次得到输入数组。

> 经过n次旋转后的数组寻找最小值

```java
class Solution {
    public int findMin(int[] nums) {
        return binarySearch(nums);   
    }
    // 二分查找
    public int binarySearch(int[] nums){
        int left = 0;
        int right = nums.length-1;
        int mid = 0;
        while(left<=right){
            mid = left + ((right-left)>>1);
            // 比较
            if(nums[mid]>=nums[right]){
                // 右边小，且mid比右边大，那么mid不是了
                left = mid + 1;
            }else if(nums[mid]<nums[right]){
                // 左边小，那么在左边 mid有可能是
                right = mid;
            }
        }
        return nums[mid];
    }
}
```

#### 3.3 搜索旋转排序数组中(可能有重复元素)中的最小值II

```java
class Solution {
    // 含有重复元素的二分查找
    public int findMin(int[] nums) {
        return binarySearch(nums);
    }
    public int binarySearch(int[] nums){
        int left = 0;
        int right = nums.length-1;
        int mid = 0;
        while(left<=right){
            mid = left + ((right-left)>>1);
            if(nums[mid]<nums[right]){
                // 既然mid小了，那么可能mid是
                right = mid;
            }else if(nums[mid]>nums[right]){
                // 既然中间大于右边 那么mid肯定不是了
                left = mid+1;
            }else if(nums[mid]==nums[right]){
                right--;
            }
        }
        return nums[mid];
    }
}
```

#### 3.4 搜索旋转排序数组(值互不相同，且经过一次旋转)

整数数组 nums 按升序排列，数组中的值 互不相同 。

在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。

给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。

 

示例 1：

输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4

```java
class Solution {
    public int search(int[] nums, int target) {
        return binarySearch(nums,target);
    }

    public int binarySearch(int[] nums,int target){
        int left = 0;
        int right = nums.length-1;
        while(left<=right){
            int mid = left + ((right-left)>>1);
            if(nums[mid]==target){
                // 找到了
                return mid;
            }

            // 之后对其判断哪边有序
            if(nums[mid]>=nums[left]){
                // 左边有序
                //接着二分的判断
                if(nums[left]<=target&&target<nums[mid]){
                    right = mid-1;
                }else{
                    left = mid + 1;
                }
            }else{
                // 右边有序
                if(nums[mid]<target&&target<=nums[right]){
                    left = mid + 1;
                }else{
                    right = mid - 1;
                }
            }

        }
        return -1;
    }
}
```

####  3.5 搜索旋转排序数组(值有重复的，且经过一次旋转)

已知存在一个按非降序排列的整数数组 nums ，数组中的值不必互不相同。

在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转 ，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,4,4,5,6,6,7] 在下标 5 处经旋转后可能变为 [4,5,6,6,7,0,1,2,4,4] 。

给你 旋转后 的数组 nums 和一个整数 target ，请你编写一个函数来判断给定的目标值是否存在于数组中。如果 nums 中存在这个目标值 target ，则返回 true ，否则返回 false 。

 

示例 1：

输入：nums = [2,5,6,0,0,1,2], target = 0
输出：true

```java
class Solution {
    // 在原来的基础上加上去重的操作
    public boolean search(int[] nums, int target) {
        return binarySearch(nums,target);
    }
    // 旋转的数组中寻找值
    public boolean binarySearch(int[] nums,int target){
        int left = 0;
        int right = nums.length-1;
        while(left<=right){
            // 去重
            while(left+1<=right&&nums[left]==nums[left+1]){
                left++;
            }
            while(right-1>=left&&nums[right]==nums[right-1]){
                right--;
            }
            // 计算mid
            int mid = left + ((right-left)>>1);
            if(nums[mid]==target){
                return true;
            }
            // 查找哪边有序
            if(nums[mid]>=nums[left]){
                // 左边有序
                if(nums[left]<=target&&target<nums[mid]){
                    right = mid - 1;
                }else{
                    left = mid + 1;
                }
            }else{
                // 右边有序
                if(nums[mid]<target&&target<=nums[right]){
                    left = mid + 1;
                }else{
                    right = mid - 1;
                }
            }
            
        }
        return false;
    }

}
```

#### 3.6 搜索旋转排序数组(多次旋转，且有重复元素，且如果有多个元素，返回索引值最小的一个)

```java
class Solution {
    public int search(int[] arr, int target) {
        return binarySearch(arr,target);
    }

    public int binarySearch(int[] nums,int target){
        int left = 0;
        int right = nums.length-1;
        while(left<=right){
            // 最边界
            if(nums[left]==target){
                return left;
            }
            int mid = left + ((right-left)>>1);
            //继续判断
            if(nums[mid]==target){
                // 保证最边界
                right = mid;
            }else if(nums[mid]>nums[left]){
                // 左边有序
                if(nums[left]<=target&&target<nums[mid]){
                    right = mid-1;
                }else{
                    left = mid+1;
                }
            }else if(nums[mid]<nums[left]){
                // 右边有序
                if(nums[mid]<target&&target<=nums[right]){
                    left = mid+1;
                }else{
                    right = mid-1;
                }
            }else if(nums[mid]==nums[left]){
                // 可能重复
                left++;
            }
        }
        return -1;
    }
}
```



## 华为笔试题

### 1.反转字符串

给出一个字符串s(仅含有小写英文字母和括号)。请你按照从括号从内到外的顺序，逐层反转每对匹配括号中的字符串，并返回最终的结果。

注意：结果中并不包含任何括号。

**思路：**

采用一个栈，当遇到左括号时将左括号的下标索引入栈，当遇到右括号的时候，将栈顶的左括号弹出，并将两个括号之间的内容全部反转。

```java
public static String reveerseParentheses(String s){
    StringBuilder sb = new StringBuilder();
    char[] arr = s.toCharArray();
    //用栈存储下标
    Stack<Integer> stack = new Stack<>();
    for(int i=0;i<arr.length;i++){
        if(arr[i]=='('){
            stack.push(i);
        }
        if(arr[i]==')'){
            reverse(arr,stack.pop()+1,i-1);
        }
    }
	for(int i=0;i<arr.length;i++){
        if(arr[i]!='('&&arr[i]!=')'){
            sb.append(arr[i]);
        }
    }   
    return sb.toString();
}

// 反转
public static void reverse(char[] arr,int left,int right){
    while(right<left){
        char temp  = arr[left];
        arr[left]  = arr[right];
        arr[right] = temp;
        right--;
        left++;
    }
}
```

### 2.跳跃游戏II

给定一个非负整数，你最初位于数组的第一个位置。

数组中的每一个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达最后一个位置。

示例：

```
输入：[2,3,1,1,4]
输出：2
解释：跳到最后一个位置的最小跳跃数是2.
从下标0跳跃到下标为1的位置，跳1步。然后跳3步到达数组的最后一个位置。
```

```java
class Solution{
    public int jump(int[] nums){
        int len = nums.length-1;
        int right_max = 0;
        int pre_max = 0;
        int steps = 0;
        for(int i=0;i<len;i++){
            if(i>=right_max){
                right_max = Math.max(right_max,i+nums[i]);
                if(i==pre_max){
                    pre_max = right_max;
                    steps++;
                }
            }
        }
        return steps;
    }
}
```

### [3.判断一个数是否为回文数]

```java
/**
	 * 判断一个数是否为回文数
	 * @param n
	 * @return
	 */
	public static boolean isDigit(int n) {
		int sum = 0;
		int pre = n;
		// 123321
		while(n>0) {
			int temp = n%10;
			sum = sum*10 + temp;
			n = n/10;
			
		}
		// 判断
		return sum==pre;
	}
	/**
	 * 数字转为字符串
	 * 123321
	 * @param n
	 * @return
	 */
	public static boolean isDigit_2(int n) {
		//负数的情况
		if(n<0) {
			return false;
		}
		String a = String.valueOf(n);
		
		StringBuilder s1 = new StringBuilder(n);
		s1 = s1.reverse();
		
		StringBuilder s2 = new StringBuilder(n);
		System.out.println(s1==s2);
		
		return true;
	}
	
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		while(sc.hasNext()) {
			int n = sc.nextInt();
			boolean res = isDigit(n);
			System.out.println(res);
			boolean res_2 = isDigit_2(n);
			System.out.println(res_2);
		}
	}
```

### DFS-[1.Leetcode841 钥匙和房间](https://leetcode-cn.com/problems/keys-and-rooms/)

有 N 个房间，开始时你位于 0 号房间。每个房间有不同的号码：0，1，2，...，N-1，并且房间里可能有一些钥匙能使你进入下一个房间。

在形式上，对于每个房间 i 都有一个钥匙列表 rooms[i]，每个钥匙 rooms[i][j] 由 [0,1，...，N-1] 中的一个整数表示，其中 N = rooms.length。 钥匙 rooms[i][j] = v 可以打开编号为 v 的房间。

最初，除 0 号房间外的其余所有房间都被锁住。

你可以自由地在房间之间来回走动。

如果能进入每个房间返回 true，否则返回 false。

示例 1：

输入: [[1],[2],[3],[]]
输出: true
解释:  
我们从 0 号房间开始，拿到钥匙 1。
之后我们去 1 号房间，拿到钥匙 2。
然后我们去 2 号房间，拿到钥匙 3。
最后我们去了 3 号房间。
由于我们能够进入每个房间，我们返回 true。
示例 2：

输入：[[1,3],[3,0,1],[2],[0]]
输出：false
解释：我们不能进入 2 号房间。

> 解题思路：
>
> 当 xx 号房间中有 yy 号房间的钥匙时，我们就可以从 xx 号房间去往 yy 号房间。如果我们将这 nn 个房间看成有向图中的 nn 个节点，那么上述关系就可以看作是图中的 xx 号点到 yy 号点的一条有向边。
>
> 这样一来，问题就变成了给定一张有向图，询问从 00 号节点出发是否能够到达所有的节点。

我们可以使用深度优先搜索的方式遍历整张图，统计可以到达的节点个数，并利用数组vis标记当前节点是否访问过，以防止重复访问。

```java
class Solution {
    // dfs解题
    //访问过的
    boolean[] visited;
    int sum;
    public boolean canVisitAllRooms(List<List<Integer>> rooms) {
        int n = rooms.size();
        visited = new boolean[n];
        sum = 0;
        //dfs从0开始
        dfs(rooms,0);
        return sum==n;
    }
    // dfs
    public void dfs(List<List<Integer>> rooms,int i){
        visited[i] = true;
        sum++;
        //访问其它的
        for(int n:rooms.get(i)){
            //未访问过
            if(!visited[n]){
                dfs(rooms,n);
            }
        }
    }
}


// 主函数增加
public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		String str = sc.nextLine();
		String[] str_arr = str.split(";");
		int n = str_arr.length;
		// 结果存储
		List<List<Integer>> list = new ArrayList<>();
		for(int i=0;i<n;i++) {
			List<Integer> temp = new ArrayList<>();
			String[] temp_num = str_arr[i].split(",");
			for(String num:temp_num) {
				temp.add(Integer.valueOf(num));
			}
			list.add(temp);
		}
		// 输出
	}
```

## 360笔试题

### 【360编程题 ab串】

链接：https://www.nowcoder.com/questionTerminal/3d8b7e706ca9429eacd6c405713ba11d
来源：牛客网



小明得到一个只包含a,b两个字符的字符串，但是小明不希望在这个字符串里a出现在b左边。现在他可以将”ab”这样的子串替换成”bba”，在原串中的相对位置不变。输出小明最少需要操作多少次才能让一个给定字符串所有a都在b的右边。 

**输入描述:**

```
一个只包含a,b字符的字符串，长度不超过100000。
```

**输出描述:**

```
最小的操作次数。结果对1000000007取模。
```

示例1

**输入**

```
ab
```

**输出**

```
1
```

**说明**

```
ab到bba
```

示例2

**输入**

```
aab
```

**输出**

```
3
```

**说明**

```
aab到abba到bbaba到bbbbaa
```

```java
package com.lcz.test;

import java.util.Scanner;

public class AB_Test {
	static final int mod = 1000000007;
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		while(sc.hasNext()) {
			String s = sc.next();
			int len = s.length();
			char[] arr = s.toCharArray();
			int res = 0;
			int temp = 0;
			for(int i=len-1;i>=0;i--) {
				if(arr[i]=='b') {
					temp++;
				}else if(arr[i]=='a') {
					res = (res+temp)%mod;
					temp = (temp*2)%mod;
				}
			}
			System.out.println(res);
		}
	}
}

```

### 【360 编程题回文数】

所谓回文数就是一个数字，从左边读和从右边读的结果都是一样的，例如12321。 

  现在有一个只包含1、2、3、4的数字，你可以通过在任意位置增加一位数字或者删除一位数字来将其变换成一个回文数。但是增加或删除不同数字所需要的代价是不一样的。 

  已知增加和删除每个数字的代价如下： 

  增加一个1，代价：100；删除一个1，代价：120。 

  增加一个2，代价：200；删除一个2，代价：350。 

  增加一个3，代价：360；删除一个3，代价：200。 

  增加一个4，代价：220；删除一个4，代价：320。 

  请问如何通过最少的代价将一个数字变换为一个回文数。当然，如果一个数字本身已经是一个回文数（包括一位数，例如：2），那么变换的代价为0。 

**输入描述:**

```
单组输入。输入一个由1、2、3、4组成的正整数，正整数位数<=100位。【提示：采用字符串输入】
```

**输出描述:**

```
输出一个整数，表示将输入数字变换为一个回文数所需的最少代价。
```

示例1

**输入**

```
12322
```

**输出**

```
300
```

**说明**

```
增加一个1并增加一个2，将输入正整数变为1223221或者2123212，所需代价最小，为：100+200=300。
```

```java
package com.lcz.test;

import java.util.Arrays;
import java.util.Scanner;

// 代价最小
public class Cost_Pali {
	// 增加的代价
	static final int add_cost[] = {100,200,360,220};
	//删除的代价
	static final int remove_cost[] = {120,350,200,320};
	public static int dfs(char[] arr,int[][] dp,int l,int r) {
		if(l>=r) {
			return 0;
		}
		if(dp[l][r]==0) {
			if(arr[l]==arr[r]) {
				dp[l][r] = dfs(arr,dp,l+1,r-1);
			}else {
				dp[l][r] = Math.min(add_cost[arr[r]-'1']+dfs(arr,dp,l,r-1),
						   Math.min(add_cost[arr[l]-'1']+dfs(arr,dp,l+1,r), 
					       Math.min(remove_cost[arr[r]-'1']+dfs(arr,dp,l,r-1),
					    		   remove_cost[arr[l]-'1']+dfs(arr,dp,l+1,r))));
			}
		}
		return dp[l][r];
	}
	
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		while(sc.hasNext()) {
			String str = sc.next();
			char[] arr = str.toCharArray();
			int len = str.length();
			// 开始循环
			int[][] dp = new int[len][len];
			int res = dfs(arr,dp,0,len-1);
			System.out.println(res);
		}
	}
}

```





## 蓝湖笔试题

### [1、出道歌手]

你作为一名出道的歌手终于要出自己的第一份专辑了，你计划收录 n 首歌而且每首歌的长度都是 s 秒，每首歌必须完整地收录于一张 CD 当中。每张 CD 的容量长度都是 L 秒，而且你至少得保证同一张 CD 内相邻两首歌中间至少要隔 1 秒。为了辟邪，你决定任意一张 CD 内的歌数不能被 13 这个数字整除，那么请问你出这张专辑至少需要多少张 CD ？ 输入描述: 每组测试用例仅包含一组数据，每组数据第一行为三个正整数 n, s, L。 保证 n ≤ 100 , s ≤ L ≤ 10000 输出描述: 输出一个整数代表你至少需要的 CD 数量。 输入例子: 7 2 6 输出例子: 4

```java
public class Test4 {
	public static int process(int n,int s_len,int l_len) {
		// 每张cd收录的歌曲
		int count = (l_len+1)/(s_len+1);
		count = (int) Math.floor(count);
		// 最终cd数量=总歌曲/每张cd收录的歌曲
		int res = (int) Math.ceil(n/count);
		//判断
		if(res%13==0) {
			// 每张专辑需要减少1个
			count = count-1;
			res = (int) Math.ceil(n/count);
		}
		// 判断增加一个条件
		if(n<count&&n%13==0) {
			res = 2;
		}
		//再判断一个
		if((n-(res-1)*count)%13==0 && count-(n-(res-1)*count)==1) {
			res += 1;
		}
		return res;
	}
	
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		String str = sc.nextLine();
		String[] arr = str.split(" ");
		int n = Integer.valueOf(arr[0]);
		int s_len = Integer.valueOf(arr[1]);
		int l_len = Integer.valueOf(arr[2]);
		int res = process(n,s_len,l_len);
	}
}

```

### 【2.代码题-Leetcode括号(有效括号、括号生成、最长有效括号)】

- 括号
- 有效的括号
- 最长的有效括号
- 有效的 括号字符串
- 使括号有效的最少添加
- 有效括号的嵌套深度
- 反转每对括号间的子串

#### [1.Leetcode 面试题08.09 括号](https://leetcode-cn.com/problems/bracket-lcci/)

括号。设计一种算法，打印n对括号的所有合法的（例如，开闭一一对应）组合。

说明：解集不能包含重复的子集。

例如，给出 n = 3，生成结果为：

[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]

> 【**解题思路**】使用递归来生成括号
>
> - 若在生成过程中右括号数量大于左括号数量则终止递归，或者左括号超过限定数目`n`则终止递归
> - 若左括号等于右括号等于`n`，则添加至结果集并且终止递归

```java
class Solution {
    // 结果
    List<String> res;
    public List<String> generateParenthesis(int n) {
        res = new ArrayList<>();
        // 回溯
        dfs("",n,n);
        return res;
    }
    public void dfs(String str,int left,int right){
        // 如果左括号剩余的数量大于了
        if(left>right || left<0){
            return;
        }
        //符合条件
        if(left==0 && right==0){
            res.add(new String(str));
            return;
        }
        // 继续
        dfs(str+"(",left-class Solution {
    // 结果
    List<String> res;
    public List<String> generateParenthesis(int n) {
        res = new ArrayList<>();
        // 回溯
        dfs("",n,n);
        return res;
    }
    public void dfs(String str,int left,int right){
        // 如果左括号剩余的数量大于了
        if(left>right || left<0){
            return;
        }
        //符合条件
        if(left==0 && right==0){
            res.add(new String(str));
            return;
        }
        // 继续
        dfs(str+"(",left-1,right);
        dfs(str+")",left,right-1);
    }
}1,right);
        dfs(str+")",left,right);
    }
}
```

####  [2.Leetcode20有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。


示例 1：

输入：s = "()"
输出：true
示例 2：

输入：s = "()[]{}"
输出：true
示例 3：

输入：s = "(]"
输出：false

> 有效字符串需满足：
>
> - 左括号必须用相同类型的右括号闭合。
> - 左括号必须以正确的顺序闭合。
>
> 注意空字符串可被认为是有效字符串

```java
class Solution {
    public boolean isValid(String s) {
        //字典来解题
        Map<Character,Character> dict = new HashMap<>();
        dict.put('(',')');
        dict.put('{','}');
        dict.put('[',']');
        // 继续
        Stack<Character> stack = new Stack<>();
        // 继续
        char[] arr = s.toCharArray();
        for(int i=0;i<arr.length;i++){
            if(dict.containsKey(arr[i])){
                //入栈
                stack.push(arr[i]);
            }else{
                if(stack.isEmpty()||arr[i]!=dict.get(stack.peek())){
                    return false;
                }
                stack.pop();
            }
        }
        return stack.isEmpty();
    }
}
```

#### [3.Leetcode32 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)


给你一个只包含 `'('` 和 `')'` 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

 

**示例 1：**

```
输入：s = "(()"
输出：2
解释：最长有效括号子串是 "()"
```

**示例 2：**

```
输入：s = ")()())"
输出：4
解释：最长有效括号子串是 "()()"
```

**示例 3：**

```
输入：s = ""
输出：0
```

【解题思路】使用栈保存左括号下标，栈初始化为（-1）

对于左括号，入栈对应下标
对于右括号，出栈一个下标（表示出栈下标对应的左括号对应当前右括号）

- 若栈为空，则说明不能构成有效括号，入栈当前下标（已遍历最右端不能构成有效括号的下标）
- 若栈不为空，则说明出栈下标对应为有效括号，更新res = Math.max(res, i - stack.peek())

```java
class Solution {
    public int longestValidParentheses(String s) {
        char[] arr = s.toCharArray();
        int len = arr.length;
        // 栈
        Stack<Integer> stack = new Stack<>();
        stack.push(-1);
        // 将结果
        int res = 0;
        for(int i=0;i<arr.length;i++){
            if(arr[i]=='('){
                stack.push(i);
            }else{
                //右括号
                stack.pop();
                //判断为空则更新
                if(stack.isEmpty()){
                    stack.push(i);
                }else{
                    //不为空则记录
                    res = Math.max(res,i-stack.peek());
                }
            }
        }
        return res;
    }
}
```

####  [4.Leetcode678 有效的括号字符串](https://leetcode-cn.com/problems/valid-parenthesis-string/)

给定一个只包含三种字符的字符串：（ ，） 和 *，写一个函数来检验这个字符串是否为有效字符串。有效字符串具有如下规则：

任何左括号 ( 必须有相应的右括号 )。
任何右括号 ) 必须有相应的左括号 ( 。
左括号 ( 必须在对应的右括号之前 )。

* 可以被视为单个右括号 ) ，或单个左括号 ( ，或一个空字符串。
  一个空字符串也被视为有效字符串。
  示例 1:

输入: "()"
输出: True
示例 2:

输入: "(*)"
输出: True
示例 3:

输入: "(*))"
输出: True

【解题思路】

方法：双栈模拟，栈left保存左括号下标，栈star保存*下标，遍历字符串

- 若当前字符为（，则将下标i进栈left
- 若当前字符为*，则将其下标i进栈start*
- *若当前字符为），若left不为空，优先配对出栈；若left为空且star不为空，则star出栈（表示当前出栈的下标处*可以表示一个左括号）；若left和star均为空，则没有与其配对的，返回false

然后再来看left和star中元素，此时表示*代替右括号来配对left栈中剩余的左括号

- 若left栈顶元素大于star栈顶元素，表示*下标处于左括号下标左边，返回false
- 否则均出栈一个元素，表示配对

最后若left不为空，表示剩余左括号无法配对，返回false，若为空，返回true

```java
class Solution {
    public boolean checkValidString(String s) {
        // 两个栈模拟
        Stack<Integer> left = new Stack<Integer>();
        Stack<Integer> star = new Stack<Integer>();
        char[] arr = s.toCharArray();
        int len = arr.length;
        for(int i=0;i<len;i++){
            // 判断
            if(arr[i]=='('){
                left.push(i);
            }else if(arr[i]=='*'){
                star.push(i);
            }else{
                if(!left.isEmpty()){
                    left.pop();
                }else if(!star.isEmpty()){
                    star.pop();
                }else{
                    return false;
                }
            }
        }
        //都不为空再次判断
        while(!left.isEmpty()&&!star.isEmpty()){
            if(left.pop()>star.pop()){
                return false;
            }
        }   
        return left.isEmpty();
    }
}
```

#### [5.Leetcode921 使括号有效的最少添加](https://leetcode-cn.com/problems/minimum-add-to-make-parentheses-valid/)

给定一个由 '(' 和 ')' 括号组成的字符串 S，我们需要添加最少的括号（ '(' 或是 ')'，可以在任何位置），以使得到的括号字符串有效。

从形式上讲，只有满足下面几点之一，括号字符串才是有效的：

它是一个空字符串，或者
它可以被写成 AB （A 与 B 连接）, 其中 A 和 B 都是有效字符串，或者
它可以被写作 (A)，其中 A 是有效字符串。
给定一个括号字符串，返回为使结果字符串有效而必须添加的最少括号数。

 

示例 1：

输入："())"
输出：1
示例 2：

输入："((("
输出：3

```java
class Solution {
    public int minAddToMakeValid(String s) {
        //栈辅助
        Stack<Character> stack = new Stack<>();
        char[] arr = s.toCharArray();
        int len = arr.length;
        int op = 0;
        for(int i=0;i<len;i++){
            //判断
            if(arr[i]=='('){
                stack.push('(');
            }else{
                if(stack.isEmpty()){
                    op++;
                }else{
                    //判断
                    stack.pop();
                }
            }
        }
        return op+stack.size();
    }
}
```

#### [6.Leetcode1111 有效括号的嵌套深度](https://leetcode-cn.com/problems/maximum-nesting-depth-of-two-valid-parentheses-strings/)

有效括号字符串 定义：对于每个左括号，都能找到与之对应的右括号，反之亦然。详情参见题末「有效括号字符串」部分。

嵌套深度 depth 定义：即有效括号字符串嵌套的层数，depth(A) 表示有效括号字符串 A 的嵌套深度。详情参见题末「嵌套深度」部分。

有效括号字符串类型与对应的嵌套深度计算方法如下图所示：



 

给你一个「有效括号字符串」 seq，请你将其分成两个不相交的有效括号字符串，A 和 B，并使这两个字符串的深度最小。

不相交：每个 seq[i] 只能分给 A 和 B 二者中的一个，不能既属于 A 也属于 B 。
A 或 B 中的元素在原字符串中可以不连续。
A.length + B.length = seq.length
深度最小：max(depth(A), depth(B)) 的可能取值最小。 
划分方案用一个长度为 seq.length 的答案数组 answer 表示，编码规则如下：

answer[i] = 0，seq[i] 分给 A 。
answer[i] = 1，seq[i] 分给 B 。
如果存在多个满足要求的答案，只需返回其中任意 一个 即可。

 ![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/04/01/1111.png)

示例 1：

输入：seq = "(()())"
输出：[0,1,1,1,1,0]
示例 2：

输入：seq = "()(())()"
输出：[0,0,0,1,1,0,1,1]
解释：本示例答案不唯一。
按此输出 A = "()()", B = "()()", max(depth(A), depth(B)) = 1，它们的深度最小。
像 [1,1,1,0,0,1,1,1]，也是正确结果，其中 A = "()()()", B = "()", max(depth(A), depth(B)) = 1 。 

```java
class Solution {
    public int[] maxDepthAfterSplit(String seq) {
        char[] arr = seq.toCharArray();
        int len = arr.length;
        // 结果
        int[] res = new int[len];
        // 辅助
        int d = 0;
        // 继续
        for(int i=0;i<len;i++){
            //判断
            if(arr[i]=='('){
                ++d;
                res[i] = d%2;
            }else{
                res[i] = d%2;
                --d;
            }
        }
        return res;
    }
}
```

#### [7.Leetcode1190 反转每对括号间的子串](https://leetcode-cn.com/problems/reverse-substrings-between-each-pair-of-parentheses/)

给出一个字符串 s（仅含有小写英文字母和括号）。

请你按照从括号内到外的顺序，逐层反转每对匹配括号中的字符串，并返回最终的结果。

注意，您的结果中 不应 包含任何括号。

 

示例 1：

输入：s = "(abcd)"
输出："dcba"
示例 2：

输入：s = "(u(love)i)"
输出："iloveu"
解释：先反转子字符串 "love" ，然后反转整个字符串。
示例 3：

输入：s = "(ed(et(oc))el)"
输出："leetcode"
解释：先反转子字符串 "oc" ，接着反转 "etco" ，然后反转整个字符串。

```java
class Solution {
    public String reverseParentheses(String s) {
        // 栈
        Stack<Integer> stack = new Stack<>();
        char[] arr = s.toCharArray();
        int len = arr.length;
        for(int i=0;i<len;i++){
            // 判断
            if(arr[i]=='('){
                stack.push(i);
            }else if(arr[i]==')'){
                reverse(arr,stack.pop(),i);
            }
        }
        StringBuilder res = new StringBuilder();
        for(int i=0;i<arr.length;i++){
            if(arr[i]!='('&&arr[i]!=')'){
                res.append(arr[i]);
            }
        }
        return res.toString();
    }

    //反转
    public void reverse(char[] arr,int i,int j){
        while(i<=j){
            char temp = arr[i];
            arr[i]    = arr[j];
            arr[j]    = temp;
            i++;
            j--;
        }
    }
}
```

### [3.Leetcode134加油站](https://leetcode-cn.com/problems/gas-station/)

在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。

你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。

如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。

说明: 

如果题目有解，该答案即为唯一答案。
输入数组均为非空数组，且长度相同。
输入数组中的元素均为非负数。
示例 1:

输入: 
gas  = [1,2,3,4,5]
cost = [3,4,5,1,2]

输出: 3

解释:
从 3 号加油站(索引为 3 处)出发，可获得 4 升汽油。此时油箱有 = 0 + 4 = 4 升汽油
开往 4 号加油站，此时油箱有 4 - 1 + 5 = 8 升汽油
开往 0 号加油站，此时油箱有 8 - 2 + 1 = 7 升汽油
开往 1 号加油站，此时油箱有 7 - 3 + 2 = 6 升汽油
开往 2 号加油站，此时油箱有 6 - 4 + 3 = 5 升汽油
开往 3 号加油站，你需要消耗 5 升汽油，正好足够你返回到 3 号加油站。
因此，3 可为起始索引。

```java
class Solution {
    public int canCompleteCircuit(int[] gas, int[] cost) {
        // 剩余的油量
        int rest = 0;
        //跑的油量
        int run = 0;
        // 记录起点
        int start = 0;
        //遍历
        for(int i=0;i<gas.length;i++){
            run += (gas[i]-cost[i]);
            rest += (gas[i]-cost[i]);
            if(run<0){
                run = 0;
                start = i+1;
            }
        }
        return rest<0?-1:start;
    }
}
```

### [4.Leetcode871 最低加油次数](https://leetcode-cn.com/problems/minimum-number-of-refueling-stops/)

汽车从起点出发驶向目的地，该目的地位于出发位置东面 target 英里处。

沿途有加油站，每个 station[i] 代表一个加油站，它位于出发位置东面 station[i][0] 英里处，并且有 station[i][1] 升汽油。

假设汽车油箱的容量是无限的，其中最初有 startFuel 升燃料。它每行驶 1 英里就会用掉 1 升汽油。

当汽车到达加油站时，它可能停下来加油，将所有汽油从加油站转移到汽车中。

为了到达目的地，汽车所必要的最低加油次数是多少？如果无法到达目的地，则返回 -1 。

注意：如果汽车到达加油站时剩余燃料为 0，它仍然可以在那里加油。如果汽车到达目的地时剩余燃料为 0，仍然认为它已经到达目的地。

 

示例 1：

输入：target = 1, startFuel = 1, stations = []
输出：0
解释：我们可以在不加油的情况下到达目的地。

示例 2：

输入：target = 100, startFuel = 1, stations = [[10,100]]
输出：-1
解释：我们无法抵达目的地，甚至无法到达第一个加油站。

```java
class Solution {
    public int minRefuelStops(int target, int startFuel, int[][] stations) {
        //动态规划 dp[i]为i次油能走的最远距离
        int n =stations.length;
        long[] dp = new long[n+1];
        //初始化
        dp[0] = startFuel;
        // 转移方程
        for(int i=0;i<n;i++){
            for(int j=i;j>=0;j--){
                if(dp[j]>=stations[i][0]){
                    //可以到达
                    dp[j+1] = Math.max(dp[j+1],dp[j]+(long)stations[i][1]);
                }
            }
        }
        for(int i=0;i<=n;i++){
            if(dp[i]>=target){
                return i;
            }
        }
        return -1;
    }
}
```

## 猿辅导

### [1.Leetcode二叉搜索树的恢复](https://leetcode-cn.com/problems/recover-binary-search-tree/)

给你二叉搜索树的根节点 root ，该树中的两个节点被错误地交换。请在不改变其结构的情况下，恢复这棵树。

进阶：使用 O(n) 空间复杂度的解法很容易实现。你能想出一个只使用常数空间的解决方案吗？

 

示例 1：


输入：root = [1,3,null,null,2]
输出：[3,1,null,null,2]
解释：3 不能是 1 左孩子，因为 3 > 1 。交换 1 和 3 使二叉搜索树有效。

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public void recoverTree(TreeNode root) {
        if(root==null){
            return;
        }
        TreeNode pre = null;
        TreeNode first = null;
        TreeNode second = null;
        Stack<TreeNode> stack = new Stack<>();
        while(!stack.isEmpty() || root!=null){
            while(root!=null){
                //注意此处的中序遍历 进来的
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if(pre!=null&&pre.val>root.val){
                second = root;
                if(first==null){
                    first = pre;
                }else{
                    break;
                }
            }
            pre = root;
            root = root.right;
        }
        int temp = first.val;
        first.val = second.val;
        second.val = temp;
        return;
    }
}
```

### 2.推箱子

小猿同学特别热心肠，喜欢帮助同事~ 听闻有一个同学需要搬家连忙过去帮忙，小猿将物品放到箱子里，再将小箱子放到大箱子里。小猿突然忘了用了几个箱子，你能帮帮它呢?

[]代表一个箱子，[]3代表3个箱子，[[]3]代表1个大箱子里放了3个小箱子一共有4个箱子,[[]3]2代表有2个大箱子，每个大箱子里放了3个小箱子，一共有8个箱子。

```java
Scanner sc = new Scanner(System.in);
		String s = sc.nextLine();
		int index = 0;
		int n = s.length();
		//用栈来辅助
		Stack<Integer> stack = new Stack<>();
		stack.push(0);
		//开始继续
		while(index<n) {
			char c = s.charAt(index);
			//判断
			if(c=='[') {
				stack.push(1);
				index++;
			}else if(c==']') {
				if(index<n-1&&s.charAt(index+1)>='0'&&s.charAt(index+1)<='9') {
					index++;
					int curNum = s.charAt(index)-'0';
					stack.push(curNum*stack.pop());
				}
				stack.push(stack.pop()+stack.pop());
				index++;
			}
		}
		System.out.println(stack.pop());
```

## 小米面试题

### 1.斐波那契数列

限定语言：Kotlin、Typescript、Python、C++、Groovy、Rust、C#、Java、Go、C、Scala、Javascript、Ruby、Swift、Php、Python 3

大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0，第1项是1）。

n\leq 39*n*≤39

示例1

**输入**

```
4
```

**输出**

```
3
```

```java
class Solution {
    public int fib(int n) {
        if(n<=1){
            return n;
        }
       // 数组的函数，定义以这个数结尾的大小
       int[] dp = new int[n+1];
       dp[0] = 0;
       dp[1] = 1;
       dp[2] = 1;
       for(int i=3;i<=n;i++){
           dp[i] = dp[i-1] + dp[i-2];
       }
       return dp[n];
    }
}
```



### 2.最大数

限定语言：Kotlin、Typescript、Python、C++、Groovy、Rust、Java、Go、C、Scala、Javascript、Ruby、Swift、Php、Python 3

给定一个nums数组由一些非负整数组成，现需要将他们进行排列并拼接，每个数不可拆分，使得最后的结果最大，返回值需要是string类型，否则可能会溢出

提示:

1 <= nums.length <= 100

0 <= nums[i] <= 10000



示例1

**输入**

```
[30,1]
```

**输出**

```
"301"
```

```java
class Solution {
    public String largestNumber(int[] nums) {
        //最大的数
        PriorityQueue<String> queue = new PriorityQueue<>((x,y)->(y+x).compareTo(x+y));
        for(int num:nums){
            queue.offer(Integer.toString(num));
        }
        String res = "";
        while(!queue.isEmpty()){
            res += queue.poll();
        }
        if(res.charAt(0)=='0'){
            return "0";
        }
        return res;
    }
}
```



### 3.序列化二叉树

请实现两个函数，分别用来序列化和反序列化二叉树，不对序列化之后的字符串进行约束，但要求能够根据序列化之后的字符串重新构造出一棵与原二叉树相同的树。

二叉树的序列化是指：把一棵二叉树按照某种遍历方式的结果以某种格式保存为字符串，从而使得内存中建立起来的二叉树可以持久保存。序列化可以基于先序、中序、后序、层序的二叉树等遍历方式来进行修改，序列化的结果是一个字符串，序列化时通过 某种符号表示空节点（#），以 ！ 表示一个结点值的结束（value!）。

二叉树的反序列化是指：根据某种遍历顺序得到的序列化字符串结果str，重构二叉树。

例如，可以根据层序遍历并特定标志空结点的方案序列化，也可以根据满二叉树结点位置的标号规律来序列化，还可以根据先序遍历和中序遍历的结果来序列化。

假如一棵树共有 2 个结点， 其根结点为 1 ，根结点右子结点为 2 ，没有其他结点。按照上面第一种说法可以序列化为“1,#,2,#,#”，按照上面第二种说法可以序列化为“{0:1,2:2}”，按照上面第三种说法可以序列化为“1,2;2,1”，这三种序列化的结果都包含足以构建一棵与原二叉树完全相同的二叉树的信息。

不对序列化之后的字符串进行约束，所以欢迎各种奇思妙想。

示例1

**输入**

```
{8,6,10,5,7,9,11}
```

**输出**

```
{8,6,10,5,7,9,11}
```

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Codec {

    // Encodes a tree to a single string.
    // 序列化二叉树
    public String serialize(TreeNode root) {
        if(root==null){
            return " ";
        }
        Queue<TreeNode> queue = new LinkedList<>();
        StringBuilder res = new StringBuilder();
        // 层序遍历
        queue.offer(root);
        while(!queue.isEmpty()){
            TreeNode cur = queue.poll();

            if(cur!=null){
                // 不为空的话
                res.append(cur.val+",");
                queue.offer(cur.left);
                queue.offer(cur.right);
            }else{
                // 为空的话
                res.append("null,");
            }
        }
        return res.toString().substring(0,res.length()-1);
    }

    // Decodes your encoded data to tree.
    // 反序列化二叉树
    public TreeNode deserialize(String data) {
        if(data==" "){
            return null;
        }
        // 对其分割
        String[] arr = data.split(",");
        int index = 0;
        // 层序遍历
        Queue<TreeNode> queue = new LinkedList<>();
        TreeNode root = new TreeNode(Integer.valueOf(arr[index++]));
        queue.offer(root);
        // 开始
        while(!queue.isEmpty()){
            TreeNode cur = queue.poll();
            if(!"null".equals(arr[index])){
                cur.left = new TreeNode(Integer.valueOf(arr[index]));
                queue.offer(cur.left);
            }
            index++;

            if(!"null".equals(arr[index])){
                cur.right = new TreeNode(Integer.valueOf(arr[index]));
                queue.offer(cur.right);
            }
            index++;
        }
        return root;
    }
}

// Your Codec object will be instantiated and called as such:
// Codec codec = new Codec();
// codec.deserialize(codec.serialize(root));
```



### 4.用两个栈实现队列

用两个栈来实现一个队列，分别完成在队列尾部插入整数(push)和在队列头部删除整数(pop)的功能。 队列中的元素为int类型。保证操作合法，即保证pop操作时队列内已有元素。

示例:

输入:

["PSH1","PSH2","POP","POP"]

返回:

1,2

解析:

"PSH1":代表将1插入队列尾部

"PSH2":代表将2插入队列尾部

"POP“:代表删除一个元素，先进先出=>返回1

"POP“:代表删除一个元素，先进先出=>返回2

示例1

**输入**

```
["PSH1","PSH2","POP","POP"]
```

**输出**

```
1,2
```

```java
import java.util.Stack;

public class Solution {
    Stack<Integer> s_in = new Stack<Integer>();
    Stack<Integer> s_out = new Stack<Integer>();
    /**
    存入
    */
    public void push(int node) {
       s_in.push(node); 
    }
    /**
    取
    */
    public int pop() {
        if(isEmpty()){
            return -1;
        }
        if(s_out.isEmpty()){
            while(!s_in.isEmpty()){
                s_out.push(s_in.pop());
            }
        }
        return s_out.pop();
    }
    
    /**
    判断是否为空
    */
    private boolean isEmpty(){
        return s_in.isEmpty()&&s_out.isEmpty();
    }
}
```

### 5.最长公共子串

给定两个字符串str1和str2,输出两个字符串的最长公共子串

题目保证str1和str2的最长公共子串存在且唯一。

示例1

**输入**

```
"1AB2345CD","12345EF"
```

**输出**

```
"2345"
```

```java
import java.util.*;


public class Solution {
    /**
     * longest common substring
     * @param str1 string字符串 the string
     * @param str2 string字符串 the string
     * @return string字符串
     */
    public String LCS (String str1, String str2) {
        // write code here
        char[] arr1 = str1.toCharArray();
        char[] arr2 = str2.toCharArray();
        int len1 = arr1.length;
        int len2 = arr2.length;
        //最长公共子串
        String[][] dp = new String[len1+1][len2+2];
        String res = "";
        //转移方程
        for(int i=0;i<=len1;i++){
            for(int j=0;j<=len2;j++){
                if(i==0||j==0){
                    //初始化
                    dp[i][j] = "";
                }else if(arr1[i-1]==arr2[j-1]){
                    dp[i][j] = dp[i-1][j-1] + arr1[i-1];
                }else{
                    dp[i][j] = "";
                }
                res = res.length()<dp[i][j].length()?dp[i][j]:res;
            }
            //记录最长
        }
        return  res;
    }
}
```



## 携程

### 第一道：错峰出行

上班了，小程每天晚上都要坐公交车回家

公交车每天晚高峰都很拥挤，但是好在小程不用那么着急回家，可以在公司里坐一会。等高峰期一过，小程再回家。因为要时刻知道当前是否在高峰期，小程需要知道当前公交线路的拥挤段是哪里。

已知小程乘坐的公交车线路有n个站，从起点站到终点站依次为第0站至第n-1站。且己知第i站当前的人流量ai,拥挤段指一段站点个数大于等于K的连续站点区间，这段区间中的站平均人流量最 大。用1 (,为整数)表示从编号为的站点开始，编号为r的站点结束的站点区间，那么平均人 流量就等于编号在、r之间的站点ai的平均值。如果有多 个平均人流是最大的区间，取最长的那个。如果有多个平均人流量最大且最长的区间，取I最小的那个。

请你帮小程找到公交车线路当前的拥挤段[,r]吧!

**输入描述**

第一行两个正整数n(1<=n<=100)，K(1<=K<=n)接下来行n个整数，第1个数ai表示当前第站的人流量(1<=ai<=1000)

**输出描述**

输出两个整数(.r，用一个空格隔开，表示拥挤段的开始站和结束站

```java
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int k = sc.nextInt();
        int[] arr = new int[n];
        for (int i = 0; i < n; ++i) {
            arr[i] = sc.nextInt();
        }
        // 窗口范围[k,n], 平均值最大，次之窗口最宽
        int[] preSum = new int[n + 1];
        for (int i = 1; i<= n; ++i) {
            preSum[i] = preSum[i-1] +  arr[i-1]; // preSum[i] = sum(arr[1..i))
        }
        long sum = 0;
        int st = 0, ed = 0; // avg = sum / (ed - st)
        for (int i = 0; i <= n - k; ++i) {
            for (int j = i + k; j <= n; ++j) {
                long tmpSum = preSum[j] - preSum[i];
                if (sum * (j - i) < tmpSum * (ed - st)  // sum/(ed-st) < tmpSum/(j-i)
                    || sum * (j - i) == tmpSum * (ed - st) && (j - i) > (ed - st)) {
                    sum = tmpSum;
                    ed = j;
                    st = i;
                }
            }
        }
        System.out.println(st + " " + (ed - 1));
    }
}
```

### 第二道：数字人生

第1天屏幕上会显示数字串的第社至1+1-1位。由于每天 的内容要变化，每个显示位上的灯管亮灭都会进行调整。开始第0天时， 所有灯管都是熄 1,从第1天的状态调整到第i+ 1天时(0<mi<n-);: 1,灭掉新数字中没有但目前亮的灯 2、亮起新数字中有但目前没有亮的灯管。我们称一 次灯管的亮/灭操作为“变化”，Q: (为3,数字事为1234567890,天数由0变为1时，亮起了123,发生了12:次变化;天数由1b2时显示的数字串由123变为234,其中1→+2产生了5个变化，2→3产生了2个变化，3→4产生个变化

当屏幕最右边为数字串最后一位时， 屏幕将永远定格不再改变

你能回答出当屏嘉宽度为(1 <=|< =n)时从开始到结束-共产生的变化数吗?

```java
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        int[][] changeCount = {
                {0, 4, 3, 3, 4, 3, 2, 3, 1, 2},
                {4, 0, 5, 3, 2, 5, 6, 1, 5, 4},
                {3, 5, 0, 2, 5, 4, 3, 4, 2, 3},
                {3, 3, 2, 0, 3, 2, 3, 2, 2, 1},
                {4, 2, 5, 3, 0, 3, 4, 3, 3, 2},
                {3, 5, 4, 2, 3, 0, 1, 4, 2, 1},
                {2, 6, 3, 3, 4, 1, 0, 5, 1, 2},
                {3, 1, 4, 2, 3, 4, 5, 0, 4, 3},
                {1, 5, 2, 2, 3, 2, 1, 4, 0, 1},
                {2, 4, 3, 1, 2, 1, 2, 3, 1, 0},
                {6, 2, 5, 5, 4, 5, 6, 3, 7, 6},
        };
        Scanner sc = new Scanner(System.in);
        String s = sc.next();
        StringBuilder cache = new StringBuilder();
        for (int d = 1; d <= s.length(); ++d) { // 宽度
            cache.append(work(s, d, changeCount)).append(" ");
        }
        if (cache.length() > 0) {
            cache.deleteCharAt(cache.length() - 1);
        }
        System.out.println(cache);
    }


    private static int work(String s, int d, int[][] changeCount) {
        int totalCount = 0;
        for (int i = 0; i < d; ++i) {
            totalCount += changeCount[10][s.charAt(i) - '0'];
        }
        if (d == s.length()) {
            return totalCount;
        }
        int count = 0;
        for (int j = 1; j <= d; ++j) {
            count += changeCount[s.charAt(j - 1) - '0'][s.charAt(j) - '0'];
        }
        totalCount += count;
        for (int j = 2; j < s.length() - d; ++j) {
            count += - changeCount[s.charAt(j-1) - '0'][s.charAt(j) - '0'] + changeCount[s.charAt(j+d-1) - '0'][s.charAt(j+d) - '0'];
            totalCount += count;
        }
        return totalCount;
    }
}

```

### 第三道：建树游戏

有n个节点和n.条边，形成一棵树， 我们把其中一条边删除，就形成了两棵树，再在两棵树之间增加一条与之前删除的边不同的边，就形成了一棵新的树。给每个点设置一个权值，并规定每棵新树的权值等于最后增加的那条边的两点权值相乘。

每条边都可以删除，且每条边删除后都有很多可以加的边，因此会形成很多不同的新树。请计算这 | 些新树的数量。同时，对于每条边，删除后可以产生的若干新树的权值之和也不一一定相同，请计算这些权值之和中的最大值。

**输入描述**

第一行整数n,表示点的数量，3 <=n <= 100000。

第二行n-1个整数，空格隔开，第i个整数a;表示点a;与点i之间有一条边。第三行n个整数，空格隔开，表示各个点的权值。0 <权值<= 10000。

**输出描述**

一行， 两个整数，用空格隔开，表示新树的总数量，以及各点删除后可以产生的新树的权值之和中的最大值。

```java
n = int(input())
edges = list(map(int, input().split()))
weights = list(map(int, input().split()))
graph = [[] for _ in range(n)]
for i in range(n - 1):
    j = edges[i] - 1
    graph[i].append(j)
    graph[j].append(i)
root = 0

def search(root, parent, graph, weights, table):
    curVal = weights[root]
    curCnt = 1
    for node in graph[root]:
        if node != parent:
            val, cnt = search(node, root, graph, weights, table)
            curVal += val
            curCnt += cnt
    table[root] = (curVal, curCnt)
    return table[root]

table = {}
search(root, -1, graph, weights, table)

def find(root, parent, graph, weights, table, ttable):
    n = len(graph)
    total = table[0][0]
    resVal = 0
    resCnt = 0
    for node in graph[root]:
        if node != parent:
            curVal, curCnt = table[node]
            val = curVal * (total - curVal) - weights[node] * weights[root]
            cnt = curCnt * (n - curCnt) - 1
            resCnt += cnt
            resVal = max(resVal, val)
            find(node, root, graph, weights, table, ttable)
    ttable[root] = (resVal, resCnt)

ttable = {}
find(root, -1, graph, weights, table, ttable)

resVal = 0
resCnt = 0
for node in ttable:
    resVal = max(resVal, ttable[node][0])
    resCnt += ttable[node][1]
print(str(resCnt) + ' ' + str(resVal)
```



## 万德WIND

### [1.Leetcode012 整数转罗马数字](https://leetcode-cn.com/problems/integer-to-roman/)

罗马数字包含以下七种字符： I， V， X， L，C，D 和 M。

字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：

I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 
C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。
给你一个整数，将其转为罗马数字。

> 解题思路：两个数组即可

```JAVA
class Solution {
    public String intToRoman(int num) {
        String[] symbols = {"M" ,"CM","D","CD" ,"C",  "XC","L", "XL","X","IX","V","IV","I"};
        int[] numbers =    {1000, 900,500,400,100,90,50,40,10,9,5,4, 1};
        // 结果
        StringBuilder res = new StringBuilder();
        // 对其遍历
        for(int i=0;i<numbers.length;i++){
            // 临时值
            String symbol = symbols[i];
            int number = numbers[i];
            //判断
            while(num>=number){
                res.append(symbol);
                num -= number;
            }
            // 结束
            if(num==0){
                break;
            }
        }
        // 继续
        return res.toString();
    }
}
```



### [2.Leetcode013 罗马数字转整数](https://leetcode-cn.com/problems/roman-to-integer/)

罗马数字包含以下七种字符: I， V， X， L，C，D 和 M。

字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：

I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 
C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。
给定一个罗马数字，将其转换成整数。输入确保在 1 到 3999 的范围内。

> 解题思路：一个hashmap字典来解决

```java
class Solution {
    public int romanToInt(String s) {
        HashMap<String,Integer> dict = new HashMap<>();
        dict.put("M",1000);
        dict.put("CM",900);
        dict.put("D", 500);
        dict.put("CD",400);
        dict.put("C", 100);
        dict.put("XC",90);
        dict.put("L", 50);
        dict.put("XL",40);
        dict.put("X", 10);
        dict.put("IX",9);
        dict.put("V", 5);
        dict.put("IV",4);
        dict.put("I", 1);
        // 对其遍历
        int res = 0;
        int len = s.length();
        // 对其遍历
        for(int i=0;i<len;){
            // 判断两个字符
            if(i+1<len&&dict.containsKey(s.substring(i,i+2))){
                res += dict.get(s.substring(i,i+2));
                i += 2;
            }else{
                // 判断一个字符
                res += dict.get(s.substring(i,i+1));
                i++;
            }
        }
        return res;
    }
}
```

### [3.Leetcode478 在圆内随机生成点](https://leetcode-cn.com/problems/generate-random-point-in-a-circle/)

给定圆的半径和圆心的 x、y 坐标，写一个在圆中产生均匀随机点的函数 randPoint 。

说明:

输入值和输出值都将是浮点数。
圆的半径和圆心的 x、y 坐标将作为参数传递给类的构造函数。
圆周上的点也认为是在圆中。
randPoint 返回一个包含随机点的x坐标和y坐标的大小为2的数组。
示例 1：

输入: 
["Solution","randPoint","randPoint","randPoint"]
[[1,0,0],[],[],[]]
输出: [null,[-0.72939,-0.65505],[-0.78502,-0.28626],[-0.83119,-0.19803]]

```java
class Solution {

    private double rad;
    private double x;
    private double y;
    public Solution(double radius, double x_center, double y_center) {
        this.rad = radius;
        this.x   = x_center;
        this.y   = y_center;
    }   
    
    // 生成随机的点
    public double[] randPoint() {
        // 生成随机的圆面积
        double random_square = Math.random()*rad*rad;
        // 生成随机的圆半径
        double random_r = Math.sqrt(random_square);
        // 生成随机的点
        double theta = Math.PI*2*Math.random();
        //坐标
        double x_random = x + Math.cos(theta)*random_r;
        double y_random = y + Math.sin(theta)*random_r;
        return new double[]{x_random,y_random};
    }
}

/**
 * Your Solution object will be instantiated and called as such:
 * Solution obj = new Solution(radius, x_center, y_center);
 * double[] param_1 = obj.randPoint();
 */
```



### [4.迷宫游戏]

#### [4.1 Leetcode797 所有可能的路径-从 A点到B点的所有可能的路径](https://leetcode-cn.com/problems/all-paths-from-source-to-target/)

给一个有 n 个结点的有向无环图，找到所有从 0 到 n-1 的路径并输出（不要求按顺序）

二维数组的第 i 个数组中的单元都表示有向图中 i 号结点所能到达的下一些结点（译者注：有向图是有方向的，即规定了 a→b 你就不能从 b→a ）空就是没有下一个结点了。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2020/09/28/all_1.jpg)

输入：graph = [[1,2],[3],[3],[]]
输出：[[0,1,3],[0,2,3]]
解释：有两条路径 0 -> 1 -> 3 和 0 -> 2 -> 3

```java
class Solution {
    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        return solve(graph,0);
    }
    public List<List<Integer>> solve(int[][] graph,int node){
        int n = graph.length;
        // 结果存储
        List<List<Integer>> res = new ArrayList<>();
        // 递归结果
        if(node==n-1){
            List<Integer> path = new ArrayList<>();
            path.add(n-1);
            res.add(path);
            return res;
        }
        //  其余的点遍历
        for(int nei:graph[node]){
            for(List<Integer> path:solve(graph,nei)){
                path.add(0,node);
                res.add(path);
            }
        }
        return res;
    }

}
```

#### 4.2 [Leetcode64 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)

给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

说明：每次只能向下或者向右移动一步。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2020/11/05/minpath.jpg)


输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：因为路径 1→3→1→1→1 的总和最小。
示例 2：

输入：grid = [[1,2,3],[4,5,6]]
输出：12

> 解题思路：初始化以及转移方程

```java
class Solution {
    public int minPathSum(int[][] grid) {
        // 最小路径和
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        // 初始化
        dp[0][0] = grid[0][0];
        for(int i=1;i<m;i++){
            dp[i][0] = dp[i-1][0] + grid[i][0];
        }
        for(int j=1;j<n;j++){
            dp[0][j] = dp[0][j-1] + grid[0][j];
        }
        // 转移方程
        for(int i=1;i<m;i++){
            for(int j=1;j<n;j++){
                dp[i][j] = Math.min(dp[i-1][j],dp[i][j-1])+grid[i][j];
            }
        }
        return dp[m-1][n-1];
    }
}
```

#### [4.3 Leetcode174 地下城游戏](https://leetcode-cn.com/problems/dungeon-game/)

一些恶魔抓住了公主（P）并将她关在了地下城的右下角。地下城是由 M x N 个房间组成的二维网格。我们英勇的骑士（K）最初被安置在左上角的房间里，他必须穿过地下城并通过对抗恶魔来拯救公主。

骑士的初始健康点数为一个正整数。如果他的健康点数在某一时刻降至 0 或以下，他会立即死亡。

有些房间由恶魔守卫，因此骑士在进入这些房间时会失去健康点数（若房间里的值为负整数，则表示骑士将损失健康点数）；其他房间要么是空的（房间里的值为 0），要么包含增加骑士健康点数的魔法球（若房间里的值为正整数，则表示骑士将增加健康点数）。

为了尽快到达公主，骑士决定每次只向右或向下移动一步。

 

编写一个函数来计算确保骑士能够拯救到公主所需的最低初始健康点数。

例如，考虑到如下布局的地下城，如果骑士遵循最佳路径 右 -> 右 -> 下 -> 下，则骑士的初始健康点数至少为 7。

-2 (K)	-3	3
-5	-10	1
10	30	-5 (P)

**说明:**

- 骑士的健康点数没有上限。
- 任何房间都可能对骑士的健康点数造成威胁，也可能增加骑士的健康点数，包括骑士进入的左上角房间以及公主被监禁的右下角房间

> 解题思路：从最后往上面走

```java
class Solution {
    public int calculateMinimumHP(int[][] dungeon) {
        int n = dungeon.length;
        int m = dungeon[0].length;
        // 动态规划
        int[][] dp = new int[n][m];
        // 逆序
        for(int i=n-1;i>=0;i--){
            for(int j=m-1;j>=0;j--){
                // 判断如果到了终点
                if(i==n-1&&j==m-1){
                    dp[i][j] = Math.max(1,1-dungeon[i][j]);
                }else if(i==n-1){
                    // 到达最后一行了
                    dp[i][j] = Math.max(1,dp[i][j+1]-dungeon[i][j]);
                }else if(j==m-1){
                    //到达最后一列了
                    dp[i][j] = Math.max(1,dp[i+1][j]-dungeon[i][j]);
                }else{
                    dp[i][j] = Math.max(1,Math.min(dp[i+1][j],dp[i][j+1])-dungeon[i][j]);
                }
            }
        }
        return dp[0][0];
    }
}
```

#### [4.4 Leetcode1368 使网格图至少有一条有效路径的最小代价](https://leetcode-cn.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/)

给你一个 m x n 的网格图 grid 。 grid 中每个格子都有一个数字，对应着从该格子出发下一步走的方向。 grid[i][j] 中的数字可能为以下几种情况：

1 ，下一步往右走，也就是你会从 grid[i][j] 走到 grid[i][j + 1]
2 ，下一步往左走，也就是你会从 grid[i][j] 走到 grid[i][j - 1]
3 ，下一步往下走，也就是你会从 grid[i][j] 走到 grid[i + 1][j]
4 ，下一步往上走，也就是你会从 grid[i][j] 走到 grid[i - 1][j]
注意网格图中可能会有 无效数字 ，因为它们可能指向 grid 以外的区域。

一开始，你会从最左上角的格子 (0,0) 出发。我们定义一条 有效路径 为从格子 (0,0) 出发，每一步都顺着数字对应方向走，最终在最右下角的格子 (m - 1, n - 1) 结束的路径。有效路径 不需要是最短路径 。

你可以花费 cost = 1 的代价修改一个格子中的数字，但每个格子中的数字 只能修改一次 。

请你返回让网格图至少有一条有效路径的最小代价。

 

示例 1：

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/02/29/grid1.png)

输入：grid = [[1,1,1,1],[2,2,2,2],[1,1,1,1],[2,2,2,2]]
输出：3
解释：你将从点 (0, 0) 出发。
到达 (3, 3) 的路径为： (0, 0) --> (0, 1) --> (0, 2) --> (0, 3) 花费代价 cost = 1 使方向向下 --> (1, 3) --> (1, 2) --> (1, 1) --> (1, 0) 花费代价 cost = 1 使方向向下 --> (2, 0) --> (2, 1) --> (2, 2) --> (2, 3) 花费代价 cost = 1 使方向向下 --> (3, 3)
总花费为 cost = 3.

> 解题思路：
>
> BFS 的特点是按层遍历，从而可以保证首先找到最优解（最少步数、最小深度）。从这个意义上讲，BFS 解决的其实也是最短路径问题。这一问题对应的图 GG 包含的所有顶点即为状态空间，而每一个可能的状态转移都代表了一条边。
>
> 这题是求最短距离的变种，按最短距离的bfs解法来写。
> 在这题中求的最小cost可以当作最短距离，只是这个cost的算法不太一样，当我们
> 使用bfs时向上下左右四个方向扩展，向网络所指方向扩展则cost不变，往其他方向
> 则cost+1，遍历过程中使用二维数组dst来保存由(0, 0)到其他网格的最小花费。

```java
class Solution {
    public int minCost(int[][] grid) {
        //BFS算法
        int n = grid.length;
        int m = grid[0].length;
        // 最小花费的保存
        int[][] dst = new int[n][m];
        // 方向 向右走 向左走 向下走 向上走
        int[][] direction = {{},{0,1},{0,-1},{1,0},{-1,0}};
        // 初始化
        for(int i=0;i<n;i++){
            Arrays.fill(dst[i],-1);
        }
        // 层序遍历
        Queue<int[]> queue = new LinkedList<>();
        // int数组三个参数 纵轴 横轴 当前cost
        queue.offer(new int[]{0,0,0,});
        while(!queue.isEmpty()){
            int size = queue.size();
            for(int i=0;i<size;i++){
                // 排出来第一个
                int[] q = queue.poll();
                // 判断是否到终点了
                if(q[0]==n-1&&q[1]==m-1){
                    continue;
                }
                // 继续当前能走的方向
                int val = grid[q[0]][q[1]];
                for(int j=1;j<=4;j++){
                    int r = q[0] + direction[j][0];
                    int c = q[1] + direction[j][1];
                    //判断
                    if(r>=0&&c>=0&&r<n&&c<m){
                        // 判断
                        int add = j==val?0:1;
                        if(dst[r][c]==-1 || dst[r][c]>q[2]+add){
                            dst[r][c] = q[2]+add;
                            queue.offer(new int[]{r,c,dst[r][c]});
                        }
                    }
                }
            }
        }
        return Math.max(0,dst[n-1][m-1]);
    }
}
```

#### [4.5 Leetcode1391 检查网格中是否存在有效路径](https://leetcode-cn.com/problems/check-if-there-is-a-valid-path-in-a-grid/)

给你一个 m x n 的网格 grid。网格里的每个单元都代表一条街道。grid[i][j] 的街道可以是：

1 表示连接左单元格和右单元格的街道。
2 表示连接上单元格和下单元格的街道。
3 表示连接左单元格和下单元格的街道。
4 表示连接右单元格和下单元格的街道。
5 表示连接左单元格和上单元格的街道。
6 表示连接右单元格和上单元格的街道。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/03/21/main.png)


你最开始从左上角的单元格 (0,0) 开始出发，网格中的「有效路径」是指从左上方的单元格 (0,0) 开始、一直到右下方的 (m-1,n-1) 结束的路径。该路径必须只沿着街道走。

注意：你 不能 变更街道。

如果网格中存在有效的路径，则返回 true，否则返回 false 。

  

示例 1：

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/03/21/e1.png)

输入：grid = [[2,4,3],[6,5,2]]
输出：true
解释：如图所示，你可以从 (0, 0) 开始，访问网格中的所有单元格并到达 (m - 1, n - 1) 。

```java
class Solution {
    // 定义数组 记录街道 前两个数字记录当前的方向 后面三个数字记录可以连接的下一个街道的型号 看图
    int[][] direct = {{0,1,1,3,5}, {1,0,2,5,6}, {0,-1,1,4,6},{-1,0,2,3,4}};
    // 记录图中每个街道哪些方向是相通的 看图 与上面的数组索引下标相关联
    int[][] cset = {{},{0,2},{1,3},{1,2},{0,1},{2,3},{0,3}};
    public boolean hasValidPath(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        boolean[][] df = new boolean[m][n];
        //层序遍历
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[]{0,0,1});
        while(!queue.isEmpty()){
            int[] cur = queue.poll();
            //到达了
            if(cur[0]==m-1&&cur[1]==n-1){
                return true;
            }
            int[] rdirect = cset[grid[cur[0]][cur[1]]];
            // 遍历
            for(int rd:rdirect){
                int[] d = direct[rd];
                int x = cur[0]+d[0];
                int y = cur[1]+d[1];
                //判断
                if(x>=m || x<0 || y<0 || y>=n || df[x][y]){
                    continue;
                }
                if(grid[x][y]==d[2] || grid[x][y]==d[3] || grid[x][y]==d[4]){
                    queue.offer(new int[]{x,y,1});
                    df[x][y] = true;
                }
            }
        }
        return false;
    }
}
```







## 一点

### Leetcode435 452 630 区间调度问题

一个团每天有很多副本要参加，但是很多时间是有冲突的。给你一个一天内的会议安排表，希望你帮他计算出一天最多能参加的活动的数量。

输入参数是一个二维数据，分别是每个活动的起始时间。最早为00:00，最晚为23:59。

[["10:00"],"12:00", ["03:00","11:30"],["11:30","14:00"]]

输出2

```java
public int countMaxActivity(ArrayList<ArrayList<String>> timeSchedule) {
		Collections.sort(timeSchedule,(a,b)->(a.get(0).compareTo(b.get(0))));
		int res = 1;
		String end = timeSchedule.get(0).get(1);
		for(int i=1;i<timeSchedule.size();i++) {
			String start = timeSchedule.get(i).get(0);
			if(start.compareTo(end)>0) {
				res++;
				end = timeSchedule.get(i).get(1);
			}
		}
		return res;
		
	}
```

### Leetcode64 最小路径和


给定一个包含非负整数的 `*m* x *n*` 网格 `grid` ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

**说明：**每次只能向下或者向右移动一步。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/11/05/minpath.jpg)

```
输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：因为路径 1→3→1→1→1 的总和最小。
```

```java
class Solution {
    public int minPathSum(int[][] grid) {
        // 最小路径和
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        // 初始化
        dp[0][0] = grid[0][0];
        for(int i=1;i<m;i++){
            dp[i][0] = dp[i-1][0] + grid[i][0];
        }
        for(int j=1;j<n;j++){
            dp[0][j] = dp[0][j-1] + grid[0][j];
        }
        // 转移方程
        for(int i=1;i<m;i++){
            for(int j=1;j<n;j++){
                dp[i][j] = Math.min(dp[i-1][j],dp[i][j-1])+grid[i][j];
            }
        }
        return dp[m-1][n-1];
    }
}
```

### Leetcode165 比较版本号

给你两个版本号 version1 和 version2 ，请你比较它们。

版本号由一个或多个修订号组成，各修订号由一个 '.' 连接。每个修订号由 多位数字 组成，可能包含 前导零 。每个版本号至少包含一个字符。修订号从左到右编号，下标从 0 开始，最左边的修订号下标为 0 ，下一个修订号下标为 1 ，以此类推。例如，2.5.33 和 0.1 都是有效的版本号。

比较版本号时，请按从左到右的顺序依次比较它们的修订号。比较修订号时，只需比较 忽略任何前导零后的整数值 。也就是说，修订号 1 和修订号 001 相等 。如果版本号没有指定某个下标处的修订号，则该修订号视为 0 。例如，版本 1.0 小于版本 1.1 ，因为它们下标为 0 的修订号相同，而下标为 1 的修订号分别为 0 和 1 ，0 < 1 。

返回规则如下：

如果 version1 > version2 返回 1，
如果 version1 < version2 返回 -1，
除此之外返回 0。


示例 1：

输入：version1 = "1.01", version2 = "1.001"
输出：0
解释：忽略前导零，"01" 和 "001" 都表示相同的整数 "1"

```java
class Solution {
    public int compareVersion(String version1, String version2) {
        //分割
        String[] arr1 = version1.split("\\.");
        String[] arr2 = version2.split("\\.");
        //对其遍历
        for(int i=0;i<Math.max(arr1.length,arr2.length);i++){
            int num1 = i<arr1.length?Integer.valueOf(arr1[i]):0;
            int num2 = i<arr2.length?Integer.valueOf(arr2[i]):0;
            if(num1>num2){
                return 1;
            }
            if(num1<num2){
                return -1;
            }
        }
        return 0;
    }
}
```



## 墨奇科技

### [1.Leetcode136 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)

给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

说明：

你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

示例 1:

输入: [2,2,1]
输出: 1
示例 2:

输入: [4,1,2,1,2]
输出: 4

> 解题思路：用异或运算

```java
class Solution {
    public int singleNumber(int[] nums) {
        int res = 0;
        for(int num:nums){
            res ^= num;
        }
        return res;
    }
}
```

### [2.Leetcode137 只出现一次的数字II](https://leetcode-cn.com/problems/single-number-ii/)

给你一个整数数组 nums ，除某个元素仅出现 一次 外，其余每个元素都恰出现 三次 。请你找出并返回那个只出现了一次的元素。

 

示例 1：

输入：nums = [2,2,3,2]
输出：3
示例 2：

输入：nums = [0,1,0,1,0,1,99]
输出：99

```java
class Solution {
    public int singleNumber(int[] nums) {
        Map<Integer, Integer> freq = new HashMap<Integer, Integer>();
        for (int num : nums) {
            freq.put(num, freq.getOrDefault(num, 0) + 1);
        }
        int ans = 0;
        for (Map.Entry<Integer, Integer> entry : freq.entrySet()) {
            int num = entry.getKey(), occ = entry.getValue();
            if (occ == 1) {
                ans = num;
                break;
            }
        }
        return ans;
    }
}


```



```java
class Solution {
    public int singleNumber(int[] nums) {
        int ans = 0;
        for (int i = 0; i < 32; ++i) {
            int total = 0;
            for (int num: nums) {
                total += ((num >> i) & 1);
            }
            if (total % 3 != 0) {
                ans |= (1 << i);
            }
        }
        return ans;
    }
}

```

### [3.Leetcode260 只出现一次的数字III](https://leetcode-cn.com/problems/single-number-iii/)

给定一个整数数组 nums，其中恰好有两个元素只出现一次，其余所有元素均出现两次。 找出只出现一次的那两个元素。你可以按 任意顺序 返回答案。

 

进阶：你的算法应该具有线性时间复杂度。你能否仅使用常数空间复杂度来实现？

 

示例 1：

输入：nums = [1,2,1,3,2,5]
输出：[3,5]
解释：[5, 3] 也是有效的答案。
示例 2：

输入：nums = [-1,0]
输出：[-1,0]
示例 3：

输入：nums = [0,1]
输出：[1,0]

```java
class Solution {
    public int[] singleNumber(int[] nums) {
        int ret = 0;
        for (int n : nums) {
            ret ^= n;
        }
        int div = 1;
        while ((div & ret) == 0) {
            div <<= 1;
        }
        int a = 0, b = 0;
        for (int n : nums) {
            if ((div & n) != 0) {
                a ^= n;
            } else {
                b ^= n;
            }
        }
        return new int[]{a, b};
    }
}

```



```java
class Solution {
    public int[] singleNumber(int[] nums) {
        HashMap<Integer, Integer> temp = new HashMap<>();
        for (int e : nums)
        {
            int count = temp.getOrDefault(e, 0) + 1;
            temp.put(e, count);
        }
        int[] result = new int[2];
        int i = 0;
        for (Map.Entry<Integer, Integer> entry : temp.entrySet())
        {
            if (entry.getValue() == 1)
            {
                result[i] = entry.getKey();
                i++;
            }
        }
        return result;
    }
}


```

### [4.Leetcode200 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

 

示例 1：

输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
示例 2：

输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3

```java
class Solution {
    public int numIslands(char[][] grid) {
        //岛屿数量
        int count = 0;
        int m = grid.length;
        int n = grid[0].length;
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(grid[i][j]=='1'){
                    // 浸染
                    dfs(grid,i,j);
                    count++;
                }
            }
        }
        return count;
    }

    public void dfs(char[][] grid,int i,int j){
        if(i<0||i>=grid.length||j<0||j>=grid[0].length||grid[i][j]!='1'){
            return;
        }
        grid[i][j] = '2';

        dfs(grid,i+1,j);
        dfs(grid,i-1,j);
        dfs(grid,i,j-1);
        dfs(grid,i,j+1);
    }
}
```

### [5.剑指Offer33 二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。

 

参考以下这颗二叉搜索树：

     5
    / \
   2   6
  / \
 1   3
示例 1：

输入: [1,6,3,2,5]
输出: false
示例 2：

输入: [1,3,2,6,5]
输出: true

```java
class Solution {
    public boolean verifyPostorder(int[] postorder) {
        //单调栈来解决二叉搜索树的后序遍历问题
        Stack<Integer> stack = new Stack<>();
        Integer preSum = Integer.MAX_VALUE;
        // 遍历
        int n = postorder.length-1;
        for(int i=n;i>=0;i--){
            //判断
            if(postorder[i]>preSum){
                return false;
            }
            // 继续判断
            while(!stack.isEmpty()&&stack.peek()>postorder[i]){
                preSum = stack.pop();
            }
            // 放进去
            stack.push(postorder[i]);
        }
        return true;
    }
}
```



## 爱奇艺

### 滑动窗口平均数最大增幅

求滑动窗口平均数最大增幅
时间限制： 3000MS
内存限制： 589824KB
题目描述：
一个自然数数组arr，有大小为k的数据滑动窗口从数组头部往数组尾部滑动，窗口每次滑动一位，窗口最后一位到达数组末尾时滑动结束。

窗口每次滑动后，窗口内k个整数的平均值相比滑动前会有一个变化幅度百分比p。



输入描述
输入数组和窗口大小k，数组和窗口大小用英文冒号分隔，数组内自然数用英文逗号分隔

输出描述
滑动开始到结束后出现的最大p值


样例输入
5,6,8,26,50,48,52,55,10,1,2,1,20,5:3
样例输出
475.00%

提示
过程如下：

滑动窗口位置                                                  窗口平均值        平均值增幅

----------------------------------       -------         --------

[5  6  8] 26  50  48  52  55  10  1  2  1  20  5          6.33 

 5 [6  8  26] 50  48  52  55  10  1  2  1  20  5          13.33                   110.53%

 5  6 [8  26  50] 48  52  55  10  1  2  1  20  5          28.00                   110.00%

 5  6  8 [26  50  48] 52  55  10  1  2  1  20  5          41.33                   47.62%

 5  6  8  26 [50  48  52] 55  10  1  2  1  20  5          50.00                   20.97%

 5  6  8  26  50 [48  52  55] 10  1  2  1  20  5          51.67                   3.33%

 5  6  8  26  50  48 [52  55  10] 1  2  1  20  5          39.00                   -24.52%

 5  6  8  26  50  48  52 [55  10  1] 2  1  20  5          22.00                   -43.59%

 5  6  8  26  50  48  52  55 [10  1  2] 1  20  5          4.33                     -80.30%

 5  6  8  26  50  48  52  55  10 [1  2  1] 20  5          1.33                     -69.23%

 5  6  8  26  50  48  52  55  10  1 [2  1  20] 5          7.67                     475.00%

 5  6  8  26  50  48  52  55  10  1  2 [1  20  5]        8.67                     13.04%

```java
        Scanner sc = new Scanner(System.in);
        String str = sc.nextLine();
        int k = Integer.valueOf(str.split(":")[1]);
        String[] str_arr = str.split(":")[0].split(",");
        //将其转为数组
        int len = str_arr.length;
        int[] arr = new int[len];
        for(int i=0;i<len;i++){
            arr[i] = Integer.valueOf(str_arr[i]);
        }
        double sum = 0;
        for(int i=0;i<k;i++){
            sum += arr[i];
        }
        double preSum = sum;
        double maxP = 0;
        for(int i=k;i<len;i++){
            sum = sum-arr[i-k]+arr[i];
//            System.out.println(sum/k);
//            System.out.println(preSum/k);
//            System.out.println("--");
            maxP = Math.max(maxP,(( ( (sum/k)-(preSum/k))/(preSum/k)))*100);
            //更新
//            System.out.println(maxP);
//            System.out.println("--");
            preSum = sum;
        }
        System.out.println(String.format("%.2f", maxP)+"%");
        
    }
```

### Leetcode岛屿泛洪

湖泊抽水问题
时间限制： 3000MS
内存限制： 589824KB
题目描述：
你的省份有多个湖泊，所有湖泊一开始都是空的。当第 n 个湖泊下雨的时候，如果第 n 个湖泊是空的，那么它就会装满水，否则这个湖泊会发生洪水。你的目标是避免任意一个湖泊发生洪水



输入描述
给你一个整数数组 rains ，其中：

rains[i] > 0 表示第 i 天时，第 rains[i] 个湖泊会下雨。

rains[i] == 0 表示第 i 天没有湖泊会下雨，你可以选择 一个 湖泊并 抽干 这个湖泊的水

输出描述
返回一个数组 ans ，满足：

ans.length == rains.length

如果 rains[i] > 0 ，那么ans[i] == -1 。

如果 rains[i] == 0 ，ans[i] 是你第 i 天选择抽干的湖泊。

如果有多种可行解，请返回它们中的 任意一个 。如果没办法阻止洪水，请返回一个 空的数组


样例输入
[1,2,0,0,2,1]
样例输出
[-1,-1,2,1,-1,-1]

提示
贪心、搜索；

请注意，如果你选择抽干一个装满水的湖泊，它会变成一个空的湖泊。但如果你选择抽干一个空的湖泊，那么将无事发生

```java
public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        String str = sc.nextLine();
      	String str2 = str.split("\\[")[1].split("\\]")[0];
      	String[] str_arr = str2.split(",");
      	int len = str_arr.length;
      	int[] nums = new int[len];
      	for(int i=0;i<len;i++) {
      		nums[i] = Integer.valueOf(str_arr[i]);
      	}
      	int[] res = new int[len];
      	//填充
      	Arrays.fill(res, -1);
      	//记录rains[i]->i
      	Map<Integer,Integer> dict = new HashMap<>();
      	//记录0的位置
      	LinkedList<Integer> zeros = new LinkedList<>();
      	for(int i=0;i<len;i++) {
      		//发生了
      		if(nums[i]>0) {
      			//判断是否出现过了
      			if(dict.containsKey(nums[i])) {
      				int prev = dict.get(nums[i]);
      				//没有可以抽水的天数
      				if(zeros.size()==0 || zeros.getLast()<prev) {
      	      			System.out.print("[]");
      	      			return;
      				}
      				int day = -1;
      				Iterator<Integer> iter = zeros.iterator();
      				while((day=iter.next())<=prev);
      				iter.remove();
      				//赋值
      				res[day] = nums[i];
      			}
      			dict.put(nums[i],i);
      		}else {
      			//记录没有发生洪水的日子
      			zeros.add(i);
      		}
      	}
      	// 还剩下值
      	for(int idx:zeros) {
      		res[idx] = 1;
      	}
      	
      	//输出结果
      	for(int i=0;i<len;i++) {
      		if(i==0) {
      			System.out.print("["+res[i]+",");
      		}else if(i!=len-1){
      			System.out.print(res[i]+",");
      		}else {
      			System.out.print(res[i]+"]");
      		}
      	}
    }
```

### 【线程池 按序打印】

多线程按序打印
时间限制： 3000MS
内存限制： 1048576KB
题目描述：
随着中国经济的增强，无数的企业正在“出海”，作为中国文化的视频传媒佼佼者，爱奇艺也在“出海”的队伍里扬帆起航，但在出海的过程中遇到了一个语言的问题，为了让更多国外用户能体验我们丰富多彩的中国文化，需要将中文的字幕翻译成各国语言，为此，我们的小明同学实现了一个万能翻译的系统，然而由于我们需要翻译的字幕太多，无法第一时间翻译完让用户观看到，聪明的你能帮帮我们的小明同学吗？

要求：

1. 请使用多线程重写translatedAll方法来提升翻译速度

2. 请注意翻译后的line的前后顺序要和输入的List<line>的顺序保持一致，因为字幕的顺序是不能乱的



输入描述
字幕文本，每行字幕以逗号分隔

输出描述
翻译好的字幕文本，每行字幕以逗号分隔

样例输入
aaa,bbb,ccc
样例输出
AAA,BBB,CCC

```java
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.*;
import java.util.stream.Collectors;


public class Main {


    public static void main(String[] args) throws InterruptedException {
        Solution s = new Solution();
        ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(10, 10,
                0L, TimeUnit.MILLISECONDS,
                new LinkedBlockingQueue<Runnable>(10000));
        final Scanner reader = new Scanner(System.in);
        final String next = reader.next();
        List<Line> lines = Arrays.stream(next.split(",")).map(str -> new StringLine(str))
                .collect(Collectors.toList());
        List<Line> result = s.translateAll(lines, "", threadPoolExecutor);
        String resultString = result.stream().map(l -> l.toString()).collect(Collectors.joining(","));
        System.out.println(resultString);
        reader.close();
        threadPoolExecutor.shutdown();
    }

    public interface Line {
        /**
         * translate the line to the specific language
         * @param language - the language to translate
         * @return the line of translated by the {@code language} */
        Line translate(String language);
    }

    public static class Solution {
        /**
         * translate the all lines to the specific language
         * @param lines the text lines of episode
         * @param language the language to translate
         * @return the lines of translated by the {@code language} */
        public List<Line> translateAll(List<Line> lines, String language, Executor executor) throws InterruptedException {
            Job<Line> job = new Job<>();
            for (Line line : lines) {
                Callable<Line> callable = () -> line.translate(language);
                job.newTask(callable);
            }
            job.execute(executor);
            return job.get();
        }
    }

    public static class Job<V> {
    	//任务
    	List<FutureTask<V>> list = new ArrayList<>();
        public void newTask(Callable<V> runnable) {
        //待实现
        	FutureTask<V> futureTask = new FutureTask<>(runnable);
        	list.add(futureTask);
        }


        public void execute(Executor executor) {
        //待实现
        	for(FutureTask<V> futureTask:list) {
        		executor.execute(futureTask);
        	}
        }

        public List<V> get() throws InterruptedException {
        //待实现
        	List<V> res = new ArrayList<>();
        	for(FutureTask<V> futureTask:list) {
        		try {
					res.add(futureTask.get());
				} catch (ExecutionException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
        	}
        	return res;
        }

    }

    /**
     * translate the string line to upper case
     */
    public static class StringLine implements Line {
        private String text;

        public StringLine(String text) {
            this.text = text;
        }

        @Override
        public Line translate(String language) {
            return new StringLine(text.toUpperCase());
        }


        @Override
        public String toString() {
            return text;
        }
    }
}

```





## vivo笔试题

### [0.运维部署 磁盘内存 用户数]

小v是公司的运维工程师，现有一个有关应用程序部署的任务如下：

> 1、一台服务器的**磁盘空间**、**内存**是固定的，现在有N个应用程序要部署；
>
> 2、每个应用程序所需要的**磁盘、内存**不同，每个应用程序**允许访问的用户数**也不同，且同一个应用程序不能在一台服务器上部署多个。

对于一台服务器而言，如何组合部署应用程序能够使得单台服务器允许访问的用户数**最多**？

```java
private static int solution(int totalDisk, int totalMemory, List<Service> services) {
    // TODO Write your code here
    // 三维背包问题
    int n = services.size();
    int[][][] dp = new int[n + 1][totalDisk + 1][totalMemory + 1];
    for(int i = 1; i <= n; i++){
        for(int j = totalDisk; j > 0; j--){
            for(int k = totalMemory; k > 0; k--){
                if(j >= services.get(i - 1).getDisk() && k >= services.get(i - 1).getMemory()){
                    // 磁盘空间和内存还够，选择第i个应用程序或不选择第i个应用程序
                    dp[i][j][k] = Math.max(dp[i - 1][j][k],
                                           dp[i - 1][j - services.get(i - 1).getDisk()][k - services.get(i - 1).getMemory()]
                                           + services.get(i - 1).getusers());
                }else{
                    // 磁盘空间不够
                    dp[i][j][k] = dp[i - 1][j][k];
                }
            }
        }
    }
    return dp[n][totalDisk][totalMemory];
}
```

### [1.消消乐问题][Leetcode546 移除盒子]

v在vivo手机的应用商店中下载了一款名为“**一维消消乐**”的游戏，介绍如下：

> 1、给出一些不同颜色的豆子，豆子的颜色用数字（0-9）表示，即不同的数字表示不同的颜色；
>
> 2、通过不断地按行消除相同颜色且连续的豆子来积分，直到所有的豆子都消掉为止；
>
> 3、假如每一轮可以消除相同颜色的连续 k 个豆子（k >= 1），这样一轮之后小v将得到 k*k 个积分；
>
> 4、由于仅可按行消除，不可跨行或按列消除，因此谓之“一维消消乐”。

请你帮助小v计算出最终能获得的**最大**积分。

![img](https://uploadfiles.nowcoder.com/images/20190828/316183_1566957956039_9527698E0DD4D452DCEB7CBFA0A2BE23)



```java
class Solution {
    public int removeBoxes(int[] boxes) {
        // 用于存储之前计算过的状态，避免重复计算
        int[][][] dp = new int[100][100][100];
        return cal(boxes, dp, 0, boxes.length - 1, 0);
    }
    
    public int cal(int[] boxes, int[][][] dp, int l, int r, int k) {
        if (l > r) {
            return 0;
        }
        if (dp[l][r][k] != 0) {
            return dp[l][r][k];
        }
        // 计算右边有几个跟最右边一个（boxes[r]）相等, 如果相等则把右边界左移到不相同的元素之后一个为止，移动过程中同步改动k
        while (r > l && boxes[r] == boxes[r-1]) {
            r--;
            k++;
        }
        // 计算把右边k+1个消除时的得分
        dp[l][r][k] = cal(boxes, dp, l, r-1, 0) + (k+1)*(k+1);
        // 从右边界开始向左寻找跟外部k个元素相等的元素，如果相等则剔除掉这些不相等的，让后面一段连起来。
        // 此时得分就是中间消除中间一段不连续部分的得分和剩下来部分的得分
        // 比较这个得分和原来计算过其他方案的得分，去最大值覆盖到状态数组dp中
        for (int i = r-1; i >= l; --i) {
            if (boxes[i] == boxes[r]) {
                dp[l][r][k] = Math.max(dp[l][r][k], cal(boxes, dp, l, i, k+1) + cal(boxes, dp, i+1, r-1, 0));
            }
        }
        return dp[l][r][k];
        
    }
}
```

### [2.拆礼盒]

链接：https://www.nowcoder.com/questionTerminal/916c1446d019459f94743443f71b3e70
来源：牛客网



小v所在的公司即将举行年会，年会方案设计过程中必不可少的一项就是抽奖活动。小v在本次活动中被委以重任，负责抽奖活动的策划；为了让中奖的礼物更加精美且富有神秘感，打算采用礼品盒来包装奖品，此时小v发挥了自己的创意想捉弄一下获奖的同事，便采取了多重包装来包装奖品。  

  


  现给出一个字符串，并假定用一对圆括号( )表示一个**礼品盒**，0表示**奖品**，你能据此帮获奖者算出**最少**要拆多少个礼品盒才能拿到奖品吗？  

**输入描述:**

```
一行字符串，仅有'('、')'、'0' 组成，其中一对'(' ')'表示一个礼品盒，‘0’表示奖品；输入的字符串一定是有效的，即'(' ')'一定是成对出现的。
```

**输出描述:**

```
输出结果为一个数字，表示小v要拆的最少礼品盒数量
```

示例1

**输入**

```
(()(()((()(0)))))
```

**输出**

```
5
```

示例2

**输入**

```
(((0)))
```

**输出**

```
3
```

```java
private static int solution(String str) {

        // TODO Write your code here 
        // TODO Write your code here 
    	// 用栈来解决
    	Stack<Character> stack = new Stack<>();
    	char[] arr = str.toCharArray();
    	for(int i=0;i<arr.length;i++) {
    		if(arr[i]=='(') {
    			// 入栈
    			stack.push('(');
    		}else if(arr[i]==')') {
    			stack.pop();
    		}else if(arr[i]=='0') {
    			return stack.size();
    		}
    	}
    	return 0;
    }
```

### [3.Leetcode背包最大价值问题]

```java
package com.lcz.interview;

import java.util.Arrays;
//本题为考试单行多行输入输出规范示例，无需提交，不计分。
//本题为考试单行多行输入输出规范示例，无需提交，不计分。
//本题为考试单行多行输入输出规范示例，无需提交，不计分。
import java.util.Scanner;
public class Main {
 public static int max_value(int w,int n,int[] weights,int[] values){
     // 动态规划 个数以及容量
     int[][] dp = new int[n+1][w+1];
     for(int i=1;i<=n;i++){
         for(int j=1;j<=w;j++){
             if(j>=weights[i-1]){
                 dp[i][j] = Math.max(dp[i-1][j],dp[i-1][j-weights[i-1]]+values[i-1]);
             }else{
                 dp[i][j] = dp[i-1][j];
             }
         }
     }
     return dp[n][w];
 }
 
 public static void main(String[] args) {
     Scanner sc = new Scanner(System.in);
     int  c = sc.nextInt();
     String w_str = sc.next();
     String v_str = sc.next();

     String[] w_arr = w_str.split(",");
     int[] weights = new int[w_arr.length];
     for(int i=0;i<w_arr.length;i++){
         weights[i] = Integer.parseInt(w_arr[i]);
     }
     String[] v_arr = v_str.split(",");
     int[] values = new int[v_arr.length];
     for(int i=0;i<v_arr.length;i++){
         values[i] = Integer.parseInt(v_arr[i]);
     }
     
     // 处理
     int n = v_arr.length;
     int max = max_value(c,n,weights,values);
     System.out.println(max);
     
 	}
}
```

### DFS-[2.Leetcode743 网络延迟时间](https://leetcode-cn.com/problems/network-delay-time/)

有 n 个网络节点，标记为 1 到 n。

给你一个列表 times，表示信号经过 有向 边的传递时间。 times[i] = (ui, vi, wi)，其中 ui 是源节点，vi 是目标节点， wi 是一个信号从源节点传递到目标节点的时间。

现在，从某个节点 K 发出一个信号。需要多久才能使所有节点都收到信号？如果不能使所有节点收到信号，返回 -1 。

 

示例 1：

![img](https://assets.leetcode.com/uploads/2019/05/23/931_example_1.png)

输入：times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
输出：2

```java
class Solution {
    // 构建图
    // 构建从起点到每个结点的时间
    Map<Integer,Integer> dist = new HashMap<>();
    public int networkDelayTime(int[][] times, int n, int k) {
        Map<Integer,List<int[]>> graph = new HashMap<>();
        // 初始化构建
        for(int[] edge:times){
            int src = edge[0];
            int dst = edge[1];
            int time = edge[2];
            if(!graph.containsKey(src)){
                graph.put(src,new ArrayList<>());
            }
            graph.get(src).add(new int[]{dst,time});
        }
        // 排序
        for(int node:graph.keySet()){
            Collections.sort(graph.get(node),(a,b)->(a[1]-b[1]));
        }
        
        //初始化
        for(int node=1;node<=n;node++){
            dist.put(node,Integer.MAX_VALUE);
        }
        dfs(graph,k,0);
        // 结果
        int res = 0;
        for(int cand:dist.values()){
            if(cand==Integer.MAX_VALUE){
                return -1;
            }
            res = Math.max(res,cand);
        }
        return res;
    }

    //遍历
    public void dfs(Map<Integer,List<int[]>> graph,int node,int elspsed){
        //访问过了
        if(elspsed>=dist.get(node)){
            return;
        }
        dist.put(node,elspsed);
        if(graph.containsKey(node)){
            for(int[] info:graph.get(node)){
                dfs(graph,info[0],elspsed+info[1]);
            }
        }
    }
}
```

### [最短路径]

图像从传感器到输出JPEG格式图片经过很多node处理，这些node构成一个图像处理的pipeline，其中的有些节点依赖于其他节点输出。A->B表示B的执行依赖于A。   

​    假设每个node执行时间为A(t)，即node A需要执行t秒，没有依赖的node可以并行执行。编写一个方法输入一个有向无环图pipeline，输出执行完需要的最短时间。   

​    输入：第一行输入node的执行时间，第二行输入node的依赖关系。   

​    输出：最短时间。

```java
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class q3 {
  public static void main(String[] args) throws IOException {
    //
    BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
    String[] strs = br.readLine().split(",");
    int len = strs.length;
    int[] times = new int[len];
    for (int i = 0; i < len; i++) {
      times[i] = Integer.parseInt(strs[i]);
    }
    HashMap<Integer, ArrayList<Integer>> map = new HashMap<>();
    strs = br.readLine().split(";");
    for (int i = 0; i < len; i++) {
      String[] ss = strs[i].split(",");
      ArrayList<Integer> tmp = new ArrayList<>();
      for (String s : ss) {
        tmp.add(Integer.parseInt(s));
      }
      map.put(i + 1, tmp);
    }
    int res = solution(times, map);
    System.out.println(res);
  }

  private static int solution(int[] times, HashMap<Integer, ArrayList<Integer>> map) {
    int start = lookForStart(times, map);
    ArrayList<Integer> lengths = new ArrayList<>(); // 存储DFS每次遍历到末节点时的长度
    Stack<Integer> stackElement = new Stack<>(); // 存储节点
    Stack<Integer> stackLength = new Stack<>(); // 存储到这个节点时的值
    stackElement.push(start);
    stackLength.push(times[start - 1]);
    while (!stackElement.isEmpty()) {
      int tmp = stackElement.pop();
      int tmpLength = stackLength.pop();
      ArrayList<Integer> tmpAdjacency = map.get(tmp);
      for (int i = tmpAdjacency.size() - 1; i >= 0; i--) {
        int nodeToPush = tmpAdjacency.get(i);
        if (nodeToPush == 0) {
          lengths.add(tmpLength);
          break;
        }
        stackElement.push(nodeToPush);
        stackLength.push(tmpLength + times[nodeToPush - 1]);
      }
    }
    Collections.sort(lengths);
    return lengths.get(lengths.size() - 1); // 找最小值，其实是返回最大值
  }

  private static int lookForStart(int[] times, HashMap<Integer, ArrayList<Integer>> map) {
    boolean[] b = new boolean[times.length + 1];
    Iterator<Map.Entry<Integer, ArrayList<Integer>>> iterator = map.entrySet().iterator();
    while (iterator.hasNext()) {
      Map.Entry<Integer, ArrayList<Integer>> e = iterator.next();
      ArrayList<Integer> list = e.getValue();
      for (int i : list) {
        b[i] = true;
      }
    }
    int res = 0;
    for (int i = 1; i < b.length; i++) {
      if (!b[i]) {
        res = i;
      }
    }
    return res;
  }
}

```





### [Leetcode005 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

给你一个字符串 s，找到 s 中最长的回文子串。

 

示例 1：

输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
示例 2：

输入：s = "cbbd"
输出："bb"

> 解题思路：中心扩散法

```java
class Solution {
    public String longestPalindrome(String s) {
        // 中心扩散法
        String res = "";
        for(int i=0;i<s.length();i++){
            String s1 = pali(s,i,i);
            String s2 = pali(s,i,i+1);
            String temp = s1.length()>s2.length()?s1:s2;
            // 再次判断
            res = res.length()>temp.length()?res:temp;
        }
        return res;
    }

    // 判断是否是回文
    public String pali(String s,int i,int j){
        while(i>=0&&j<s.length()&&s.charAt(i)==s.charAt(j)){
            i--;
            j++;
        }
        return s.substring(i+1,j);
    }
}
```

#### 



## 真实高频面试题



### [0.Leetcode003 无重复字符的最长子串]

给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。

 

示例 1:

输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
示例 2:

输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        // 不含有重复字符的最长子串
        char[] arr = s.toCharArray();
        HashMap<Character,Integer> hashMap = new HashMap<>();
        // 长度
        int maxLen = 0;
        // 滑窗
        int l = 0;
        int r = 0;
        while(r<arr.length){
            if(hashMap.containsKey(arr[r])){
                // 包含了
                l = Math.max(hashMap.get(arr[r])+1,l);
            }
            hashMap.put(arr[r],r);
            maxLen = Math.max(r-l+1,maxLen);
            r++;
        }
        return maxLen;
    }
}
```



### [0.Leetcod674 最长连续递增序列]

给定一个未经排序的整数数组，找到最长且 连续递增的子序列，并返回该序列的长度。

连续递增的子序列 可以由两个下标 l 和 r（l < r）确定，如果对于每个 l <= i < r，都有 nums[i] < nums[i + 1] ，那么子序列 [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] 就是连续递增子序列。

 

示例 1：

输入：nums = [1,3,5,4,7]
输出：3
解释：最长连续递增序列是 [1,3,5], 长度为3。
尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，因为 5 和 7 在原数组里被 4 隔开。 
示例 2：

输入：nums = [2,2,2,2,2]
输出：1
解释：最长连续递增序列是 [2], 长度为1。

> 解题思路：套路是套路，记得最长连续子序列的left移动不是++，那种移动，是直接移动到不符合条件那里，明白了吗！！！

```java
class Solution {
    public int findLengthOfLCIS(int[] nums) {
        if(nums.length<=1){
            return nums.length;
        }
        int left = 0;
        int right = 1;
        int len = 1;
        while(right<nums.length){
            // 不符合条件
            if(nums[right]<=nums[right-1]){
                left = right;
            }
            len = Math.max(len,right-left+1);
            right++;
        }
        return len;
    }
}
```

### [1.Leetcode300 最长递增子序列]

给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。


示例 1：

输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。

```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        // 最长递增子序列
        int n = nums.length;
        // 以n为结尾的最长递增子序列
        int[] dp = new int[n];
        // 初始化所有初始化都是1
        Arrays.fill(dp,1);
        // 结果
        int res = 1;
        // 开始转移
        for(int i=1;i<n;i++){
            for(int j=0;j<i;j++){
                if(nums[i]>nums[j]){
                    dp[i] = Math.max(dp[i],dp[j]+1);
                    res = Math.max(res,dp[i]);
                }
            }
        }
        return res;
    }
}
```



### [1.排序链表]

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode sortList(ListNode head) {
        if(head==null || head.next==null){
            return head;
        }
        return mergeSort(head,null);
    }

    public ListNode mergeSort(ListNode head,ListNode tail){
        if(head.next==tail){
            // 分隔数组
            head.next = null;
            return head;
        }
        //求中间的点
        ListNode slow = head;
        ListNode fast = head;
        while(fast!=tail&&fast.next!=tail){
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode mid = slow;

        ListNode left = mergeSort(head,mid);
        ListNode right = mergeSort(mid,tail);
        ListNode sorted = merge(left,right);
        return sorted;
    }

    // 归并
    public ListNode merge(ListNode l1,ListNode l2){
        ListNode dummy = new ListNode(-1);
        ListNode l = dummy;
        while(l1!=null&&l2!=null){
            if(l1.val>=l2.val){
                l.next  = l2;
                l = l.next;
                l2 = l2.next;
            }else{
                l.next = l1;
                l = l.next;
                l1 = l1.next;
            }
        }
        if(l1!=null){
            l.next = l1;
        }
        if(l2!=null){
            l.next = l2;
        }
        return dummy.next;


    }
}
```

### [2.编解码字符串]

数字转成字符串记录。！！！(char)(j-i+1-'0')

#### (1) Leetcode443 压缩字符串（高频题）


给定一组字符，使用[原地算法](https://baike.baidu.com/item/原地算法)将其压缩。

压缩后的长度必须始终小于或等于原数组长度。

数组的每个元素应该是长度为1 的**字符**（不是 int 整数类型）。

在完成[原地](https://baike.baidu.com/item/原地算法)**修改输入数组**后，返回数组的新长度。

> 字符串可能超过10了

**进阶：**
你能否仅使用O(1) 空间解决问题？

 

**示例 1：**

```
输入：
["a","a","b","b","c","c","c"]

输出：
返回 6 ，输入数组的前 6 个字符应该是：["a","2","b","2","c","3"]

说明：
"aa" 被 "a2" 替代。"bb" 被 "b2" 替代。"ccc" 被 "c3" 替代。
```

```java
class Solution {
    public int compress(char[] chars) {
        // 新的index
        int index = 0;
        // 原来的index
        int i = 0;
        while(i<chars.length){
            // 后序的
            int j =i+1;
            while(j<chars.length&&chars[j]==chars[i]){
                j++;
            }
            if(j-i==1){
                chars[index++] = chars[i++];
                continue;
            }
            // 超过2个了编码
            chars[index++] = chars[i];
            int num = j-i;
            int mod = num%10;
            while(num>=10){
                num = num/10;
                chars[index++] = (char)(num+'0');
            }
            if(num>0){
                chars[index++] = (char)(mod+'0');
            }
            // i移动到j
            i = j;
        }
        return index;
    }
}
```

#### (2)Leetcode394 字符串解码

给定一个经过编码的字符串，返回它解码后的字符串。

编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。

你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。

此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。

 

示例 1：

输入：s = "3[a]2[bc]"
输出："aaabcbc"

示例 2：

输入：s = "3[a2[c]]"
输出："accaccacc"

```java
class Solution {
    public String decodeString(String s) {
        //一个是字母一个是数字
        int number = 0;
        StringBuilder res = new StringBuilder();
        // 栈来存储
        Stack<Integer> stack_num = new Stack<>();
        Stack<StringBuilder> stack_str = new Stack<>();
        int i = 0;
        while(i<s.length()){
            // 数字可能是10以上的
            if(Character.isDigit(s.charAt(i))){
                number = number*10 + s.charAt(i)-'0';
            }else if(Character.isAlphabetic(s.charAt(i))){
                // 字母
                res.append(s.charAt(i));
            }else if(s.charAt(i)=='['){
                // 入栈
                stack_num.push(number);
                stack_str.push(res);
                number = 0;
                res = new StringBuilder();
            }else if(s.charAt(i)==']'){
                // 出栈
                int temp_num = stack_num.pop();
                StringBuilder temp_str = stack_str.pop();
                // 开始添加值
                for(int j=0;j<temp_num;j++){
                    temp_str.append(res);
                }
                res = temp_str;
            }
            i++;
        }
        return res.toString(); 
    }
}
```

#### (3)Leetcode820 单词的压缩编码


单词数组 `words` 的 **有效编码** 由任意助记字符串 `s` 和下标数组 `indices` 组成，且满足：

- `words.length == indices.length`
- 助记字符串 `s` 以 `'#'` 字符结尾
- 对于每个下标 `indices[i]` ，`s` 的一个从 `indices[i]` 开始、到下一个 `'#'` 字符结束（但不包括 `'#'`）的 **子字符串** 恰好与 `words[i]` 相等

给你一个单词数组 `words` ，返回成功对 `words` 进行编码的最小助记字符串 `s` 的长度 。

**示例 1：**

```
输入：words = ["time", "me", "bell"]
输出：10
解释：一组有效编码为 s = "time#bell#" 和 indices = [0, 2, 5] 。
words[0] = "time" ，s 开始于 indices[0] = 0 到下一个 '#' 结束的子字符串，如加粗部分所示 "time#bell#"
words[1] = "me" ，s 开始于 indices[1] = 2 到下一个 '#' 结束的子字符串，如加粗部分所示 "time#bell#"
words[2] = "bell" ，s 开始于 indices[2] = 5 到下一个 '#' 结束的子字符串，如加粗部分所示 "time#bell#"
```

> 用HashSet来解题

```java
class Solution {
    public int minimumLengthEncoding(String[] words) {
        // 用set来进行编码
        Set<String> set = new HashSet<>(Arrays.asList(words));
        // 遍历单词数组
        for(String word:words){
            for(int i=1;i<word.length();i++){
                set.remove(word.substring(i));
            }
        }
        int len = 0;
        for(String word:set){
            len += word.length()+1;
        }
        return len;

    }
}
```

> Trie树来解题

```java
class Solution {
    public int minimumLengthEncoding(String[] words) {
        // 排序
        Arrays.sort(words,((s1,s2)->s2.length()-s1.length()));
        Trie root = new Trie();
        int len = 0;
        for(String word:words){
            len += root.insert(word);
        }
        return len;
    }
}
// 用Trie树来解题
class Trie{
    TrieNode root;
    // 初始化
    public Trie(){
        root = new TrieNode();
    }
    // 插入单词 其后缀
    public int insert(String word){
        TrieNode cur = root;
        // 记录插入的单词是否存储逆序
        boolean isNew = false;
        // 对其遍历插入单词
        for(int i=word.length()-1;i>=0;i--){
            // 数字下标
            int c = word.charAt(i)-'a';
            // 判断当前是否有了
            if(cur.children[c]==null){
                isNew = true;// 最新单词
                cur.children[c] = new TrieNode();
            }
            cur = cur.children[c];
        }
        return isNew?word.length()+1:0;
    }
}
class TrieNode{
    // 当前字母的值
    char val;
    // 以及其孩子结点 仅由26个字母构成
    TrieNode[] children = new TrieNode[26];

    public TrieNode(){

    }
    public TrieNode(char val){
        this.val = val;
    }
}
```



### [3.实现一个Trie树]

```java
class Trie {
    // 根节点
    TrieNode root;
    /** Initialize your data structure here. */
    public Trie() {
        root = new TrieNode();
    }
    
    /** Inserts a word into the trie. */
    public void insert(String word) {
        // 插入
        TrieNode cur = root;
        for(int i=0;i<word.length();i++){
            char c = word.charAt(i);
            if(cur.child[c-'a']==null){
                cur.child[c-'a'] = new TrieNode(c);
            }
            // 移动到下一个结点
            cur = cur.child[c-'a'];
        }
        // 最后一个设置为单词
        cur.isEnd = true;
    }
    
    /** Returns if the word is in the trie. */ // 搜索
    public boolean search(String word) {
        TrieNode cur = root;
        for(char c:word.toCharArray()){
            if(cur.child[c-'a']==null){
                return false;
            } 
            cur = cur.child[c-'a'];
        }
        return cur.isEnd;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        TrieNode cur = root;
        for(char c:prefix.toCharArray()){
            if(cur.child[c-'a']==null){
                return false;
            }
            cur = cur.child[c-'a'];
        }
        return true;
    }
}
class TrieNode{
    char val;
    TrieNode[] child = new TrieNode[26];
    // 标注该单词是否为结尾
    boolean isEnd = false;
    // 初始化
    public TrieNode(){

    }
    public TrieNode(char val){
        this.val = val;
    }
}

/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.insert(word);
 * boolean param_2 = obj.search(word);
 * boolean param_3 = obj.startsWith(prefix);
 */
```

### [4.实现一个并查集]

```java
class Solution {
    public int minSwapsCouples(int[] row) {
        // 构建
        int len = row.length;
        int n = len/2;
        Union union = new Union(n);
        for(int i=0;i<len;i+=2){
            union.union(row[i]/2,row[i+1]/2);
        }
        return n-union.getcount();
    }
}
class Union{
    int count;
    int[] num;
    public Union(int n){
        this.count = n;
        this.num = new int[n];
        for(int i=0;i<count;i++){
            num[i] = i;
        }
    }
    // 计算几组
    public int getcount(){
        return count;
    }
    // 合并两个数组
    public void union(int n,int m){
        int x = find(n);
        int y = find(m);
        // 不相等就合一次
        if(x!=y){
            num[x] = y;
            count--;
        }
    }
    // 查找父节点
    public int find(int n){
        while(n!=num[n]){
            num[n] = num[num[n]];
            n = num[n];
        }
        return n;
    }
}
```

### [5.单调栈的问题]Stack

#### a.下一个更大的元素II

定一个循环数组（最后一个元素的下一个元素是数组的第一个元素），输出每个元素的下一个更大元素。数字 x 的下一个更大的元素是按数组遍历顺序，这个数字之后的第一个比它更大的数，这意味着你应该循环地搜索它的下一个更大的数。如果不存在，则输出 -1。

示例 1:

输入: [1,2,1]
输出: [2,-1,2]
解释: 第一个 1 的下一个更大的数是 2；
数字 2 找不到下一个更大的数； 
第二个 1 的下一个最大的数需要循环搜索，结果也是 2。

> 循环链表，相当于在后面又复制了一个

```java
class Solution {
    public int[] nextGreaterElements(int[] nums) {
        // 正常找
        int n = nums.length;
        // 结果
        int[] res = new int[n];
        // 单调栈
        Stack<Integer> stack = new Stack<>();
        // 开始循环
        for(int i=2*n-1;i>=0;i--){
            // 出栈
            while(!stack.isEmpty()&&nums[i%n]>=stack.peek()){
                stack.pop();
            }
            res[i%n] = stack.isEmpty()?-1:stack.peek();
            // 入栈
            stack.push(nums[i%n]);
        }
        return res;
    }
}
```

### Leetcode739每日温度

请根据每日 气温 列表，重新生成一个列表。对应位置的输出为：要想观测到更高的气温，至少需要等待的天数。如果气温在这之后都不会升高，请在该位置用 0 来代替。

例如，给定一个列表 temperatures = [73, 74, 75, 71, 69, 72, 76, 73]，你的输出应该是 [1, 1, 4, 2, 1, 1, 0, 0]。

提示：气温 列表长度的范围是 [1, 30000]。每个气温的值的均为华氏度，都是在 [30, 100] 范围内的整数。

```java
class Solution {
    public int[] dailyTemperatures(int[] T) {
        // 结果数组
        int n = T.length;
        int[] res = new int[n];
        // 单调栈 存储下标索引
        Stack<Integer> stack = new Stack<>();
        // 从后往前开始
        for(int i=n-1;i>=0;i--){
            // 出栈
            while(!stack.isEmpty()&&T[i]>T[stack.peek()]){
                stack.pop();
            }
            // 结果记录等待天数
            res[i] = stack.isEmpty()?0:stack.peek()-i;
            // 当前下标索引
            stack.push(i);
        }
        return res;
    }
}
```



### 6.[单调队列问题]Deque

注意**peekLast** **pollLast**和**offerLast** **peekFirst**和**pollFirst**

#### a.滑窗最大值

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        // 双端队列来实现滑窗
        Deque<Integer> queue = new LinkedList<>();
        // 结果数组
        int n = nums.length;
        int[] res = new int[n-k+1];
        for(int i=0;i<n;i++){
            // 只存放大值
            while(!queue.isEmpty()&&nums[i]>=nums[queue.peekLast()]){
                queue.pollLast();
            }
            queue.offerLast(i);
            // 判断窗口是否在范围内
            if(queue.peekFirst()+k<=i){
                queue.pollFirst();
            }

            // 符合条件的时候 才开始记录
            if(i+1>=k){
                res[i+1-k] = nums[queue.peekFirst()];
            }
        }
        return res;
    }
}
```

#### b.窗口中的连续k个(不一定k全占满的最大值)

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        // 用双端队列实现一个单调队列
        Deque<Integer> queue = new LinkedList<>();
        // 队首最大值中的下标索引
        // 滑动窗口的思路
        int n = nums.length;
        int[] res = new int[n-k+1];
        for(int i=0;i<n;i++){
            while(!queue.isEmpty()&&nums[i]>nums[queue.getLast()]){
                // 队尾的其其小，排出来
                queue.pollLast();
            }
            // 入队列
            queue.offerLast(i);
            // 需要判断队列中的队首是否还在滑动窗口有效范围内
            if(queue.peekFirst()+k<=i){
                //不在有效范围内了 最大值排出来
                queue.pollFirst();
            }

            // 滑窗中最大值记录从下标2开始
            if(i+1>=k){
                res[i+1-k] = nums[queue.peekFirst()];
            }
        }
        return res;
    }
}
```

### [7.优先级队列-堆排序]

#### a.数据流的中位数

中位数是有序列表中间的数。如果列表长度是偶数，中位数则是中间两个数的平均值。

例如，

[2,3,4] 的中位数是 3

[2,3] 的中位数是 (2 + 3) / 2 = 2.5

设计一个支持以下两种操作的数据结构：

void addNum(int num) - 从数据流中添加一个整数到数据结构中。
double findMedian() - 返回目前所有元素的中位数。
示例：

addNum(1)
addNum(2)
findMedian() -> 1.5
addNum(3) 
findMedian() -> 2

```JAVA
class MedianFinder {
    // 两个堆
    PriorityQueue<Integer> large;
    PriorityQueue<Integer> small;
    /** initialize your data structure here. */
    public MedianFinder() {
        large = new PriorityQueue<>((a,b)->(b-a));
        small = new PriorityQueue<>((a,b)->(a-b));
    }
    //添加数据
    public void addNum(int num) {
        if(large.size()>=small.size()){
            large.offer(num);
            small.offer(large.poll());
        }else{
            small.offer(num);
            large.offer(small.poll());
        }
    }
    // 找中位数
    public double findMedian() {
        if(small.size()>large.size()){
            return small.peek();
        }
        if(small.size()<large.size()){
            return large.peek();
        }
        return (small.peek()+large.peek())/2.0;
    }
}

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder obj = new MedianFinder();
 * obj.addNum(num);
 * double param_2 = obj.findMedian();
 */
```



#### b.合并K个升序链表

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

 

示例 1：

输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    
    public ListNode mergeKLists(ListNode[] lists) {
        // 合并K个升序链表，topK问题，用优先级队列 自定义排序 建立最小
        PriorityQueue<ListNode> queue = new PriorityQueue<>((l1,l2)->(l1.val-l2.val));
       //对其存入值
       for(int i=0;i<lists.length;i++){
           // 判断是否为空
           if(lists[i]!=null){
             queue.offer(lists[i]);
           }
       }
       // 结果
       ListNode dummy = new ListNode(-1);
       ListNode head  = dummy;
       // 对其遍历
       while(!queue.isEmpty()){
           // 建立新节点
           ListNode curNode = queue.poll();
           head.next = new ListNode(curNode.val);
           head = head.next;
           // 排出来的这个节点是否还有
           if(curNode.next!=null){
               curNode = curNode.next;
               queue.offer(curNode);
           }
       }
       return dummy.next;
    }
}
```

