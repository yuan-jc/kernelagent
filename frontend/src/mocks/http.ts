/**
 * mock 层内部错误：mockRequest 抛出，client.ts 统一转成 ApiError，
 * 从而与真实 HTTP 路径共享同一套错误处理（404/400/409）。
 */
export class MockHttpError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "MockHttpError";
    this.status = status;
  }
}
